"""
auth_system.py
==============

Authentication for the public scaffold interface.

Invite-code based registration — you control who gets access
by generating invite codes. Users register with a code,
then log in with email and password.

JWT tokens for session management — stateless, works across
multiple server restarts, no database needed for session storage.

Three tiers:
  admin   — can see all users, generate invite codes, view stats
  user    — full access to the scaffold
  guest   — read-only, can see insights and workspace but not ask questions

Usage:
  from auth_system import AuthSystem, require_auth, require_admin

  auth = AuthSystem(secret_key="your-secret-key")

  # In FastAPI:
  @app.post("/register")
  async def register(req: RegisterRequest):
      return auth.register(req.email, req.password, req.invite_code)

  @app.get("/protected")
  async def protected(user = Depends(require_auth)):
      return {"user": user.email}
"""

import os
import json
import time
import hashlib
import secrets
import hmac
from dataclasses import dataclass, field
from typing import Optional

try:
    import jwt
    HAS_JWT = True
except ImportError:
    HAS_JWT = False

try:
    import bcrypt
    HAS_BCRYPT = True
except ImportError:
    HAS_BCRYPT = False


# ============================================================
# User model
# ============================================================

@dataclass
class User:
    user_id:      str
    email:        str
    password_hash: str
    role:         str   = "user"      # admin | user | guest
    created_at:   float = field(default_factory=time.time)
    last_login:   float = 0.0
    n_questions:  int   = 0
    is_active:    bool  = True
    invite_code_used: str = ""

    def to_dict(self) -> dict:
        return {
            "user_id":        self.user_id,
            "email":          self.email,
            "password_hash":  self.password_hash,
            "role":           self.role,
            "created_at":     self.created_at,
            "last_login":     self.last_login,
            "n_questions":    self.n_questions,
            "is_active":      self.is_active,
            "invite_code_used": self.invite_code_used,
        }

    def public_dict(self) -> dict:
        """Safe dict without password hash."""
        d = self.to_dict()
        d.pop("password_hash", None)
        return d


# ============================================================
# AuthSystem
# ============================================================

class AuthSystem:
    """
    JWT-based authentication with invite codes.

    Stores users in a simple JSON file — no database needed.
    For small deployments (< 1000 users) this is perfectly adequate.
    """

    TOKEN_EXPIRE_HOURS = 24 * 7   # 1 week

    def __init__(
        self,
        secret_key: str  = "",
        path:       str  = "./shared_data/auth.json",
        verbose:    bool = True,
    ):
        self.secret  = secret_key or os.environ.get(
            "SCAFFOLD_SECRET", secrets.token_hex(32)
        )
        self.path    = path
        self.verbose = verbose

        self._users:        dict[str, User]   = {}
        self._invite_codes: dict[str, dict]   = {}

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._load()

        # Create default admin if no users exist
        if not self._users:
            admin_code = self.generate_invite_code(role="admin", limit=1)
            if verbose:
                print(f"[auth] No users found. Admin invite code: {admin_code}")
                print(f"[auth] Register at /register with this code.")

    # ── Registration ──────────────────────────────────────

    def register(
        self,
        email:       str,
        password:    str,
        invite_code: str,
    ) -> dict:
        """Register a new user with an invite code."""
        email = email.lower().strip()

        # Validate invite code
        code_data = self._invite_codes.get(invite_code)
        if not code_data:
            return {"success": False, "error": "Invalid invite code"}
        if not code_data.get("active", True):
            return {"success": False, "error": "Invite code already used or expired"}
        limit = code_data.get("limit", 1)
        used  = code_data.get("used", 0)
        if limit > 0 and used >= limit:
            return {"success": False, "error": "Invite code has reached its limit"}

        # Check email not already registered
        if any(u.email == email for u in self._users.values()):
            return {"success": False, "error": "Email already registered"}

        # Validate password
        if len(password) < 8:
            return {"success": False, "error": "Password must be at least 8 characters"}

        # Create user
        user_id      = secrets.token_hex(8)
        password_hash = self._hash_password(password)
        role         = code_data.get("role", "user")

        user = User(
            user_id=user_id,
            email=email,
            password_hash=password_hash,
            role=role,
            invite_code_used=invite_code,
        )
        self._users[user_id] = user

        # Mark invite code as used
        self._invite_codes[invite_code]["used"] = used + 1
        if limit > 0 and used + 1 >= limit:
            self._invite_codes[invite_code]["active"] = False

        self._save()

        token = self._create_token(user)

        if self.verbose:
            print(f"[auth] Registered: {email} ({role})")

        return {
            "success":  True,
            "token":    token,
            "user":     user.public_dict(),
        }

    # ── Login ─────────────────────────────────────────────

    def login(self, email: str, password: str) -> dict:
        """Log in and return a JWT token."""
        email = email.lower().strip()
        user  = next(
            (u for u in self._users.values() if u.email == email),
            None,
        )

        if not user:
            return {"success": False, "error": "Invalid email or password"}

        if not user.is_active:
            return {"success": False, "error": "Account suspended"}

        if not self._verify_password(password, user.password_hash):
            return {"success": False, "error": "Invalid email or password"}

        user.last_login = time.time()
        self._save()

        return {
            "success": True,
            "token":   self._create_token(user),
            "user":    user.public_dict(),
        }

    # ── Token verification ────────────────────────────────

    def verify_token(self, token: str) -> Optional[User]:
        """Verify a JWT token and return the User, or None."""
        if not HAS_JWT:
            # Fallback: simple HMAC token
            return self._verify_hmac_token(token)
        try:
            payload = jwt.decode(
                token, self.secret, algorithms=["HS256"]
            )
            user_id = payload.get("user_id")
            return self._users.get(user_id)
        except Exception:
            return None

    # ── Invite codes ──────────────────────────────────────

    def generate_invite_code(
        self,
        role:  str = "user",
        limit: int = 1,    # how many times it can be used (0 = unlimited)
    ) -> str:
        """Generate a new invite code."""
        code = secrets.token_urlsafe(12)
        self._invite_codes[code] = {
            "role":    role,
            "limit":   limit,
            "used":    0,
            "active":  True,
            "created": time.time(),
        }
        self._save()
        if self.verbose:
            print(f"[auth] Generated invite code: {code} (role={role}, limit={limit})")
        return code

    def list_invite_codes(self) -> list[dict]:
        return [
            {"code": k, **v}
            for k, v in self._invite_codes.items()
        ]

    # ── Admin ─────────────────────────────────────────────

    def list_users(self) -> list[dict]:
        return [u.public_dict() for u in self._users.values()]

    def record_question(self, user_id: str) -> None:
        user = self._users.get(user_id)
        if user:
            user.n_questions += 1
            self._save()

    def suspend_user(self, user_id: str) -> bool:
        user = self._users.get(user_id)
        if user:
            user.is_active = False
            self._save()
            return True
        return False

    def stats(self) -> dict:
        users = list(self._users.values())
        return {
            "total_users":   len(users),
            "active_users":  sum(1 for u in users if u.is_active),
            "admin_users":   sum(1 for u in users if u.role == "admin"),
            "total_questions": sum(u.n_questions for u in users),
            "invite_codes":  len(self._invite_codes),
        }

    # ── Internals ─────────────────────────────────────────

    def _hash_password(self, password: str) -> str:
        if HAS_BCRYPT:
            return bcrypt.hashpw(
                password.encode(), bcrypt.gensalt()
            ).decode()
        # Fallback: PBKDF2
        salt  = secrets.token_hex(16)
        hashed = hashlib.pbkdf2_hmac(
            "sha256", password.encode(), salt.encode(), 100000
        ).hex()
        return f"pbkdf2:{salt}:{hashed}"

    def _verify_password(self, password: str, stored: str) -> bool:
        if HAS_BCRYPT and not stored.startswith("pbkdf2:"):
            try:
                return bcrypt.checkpw(password.encode(), stored.encode())
            except Exception:
                return False
        # PBKDF2 fallback
        if stored.startswith("pbkdf2:"):
            parts  = stored.split(":")
            salt   = parts[1]
            hashed = hashlib.pbkdf2_hmac(
                "sha256", password.encode(), salt.encode(), 100000
            ).hex()
            return hmac.compare_digest(hashed, parts[2])
        return False

    def _create_token(self, user: User) -> str:
        payload = {
            "user_id": user.user_id,
            "email":   user.email,
            "role":    user.role,
            "exp":     time.time() + self.TOKEN_EXPIRE_HOURS * 3600,
        }
        if HAS_JWT:
            return jwt.encode(payload, self.secret, algorithm="HS256")
        # HMAC fallback
        return self._create_hmac_token(payload)

    def _create_hmac_token(self, payload: dict) -> str:
        data = json.dumps(payload, sort_keys=True)
        sig  = hmac.new(
            self.secret.encode(), data.encode(), hashlib.sha256
        ).hexdigest()
        import base64
        return base64.urlsafe_b64encode(
            f"{data}|{sig}".encode()
        ).decode()

    def _verify_hmac_token(self, token: str) -> Optional[User]:
        try:
            import base64
            decoded  = base64.urlsafe_b64decode(token.encode()).decode()
            data, sig = decoded.rsplit("|", 1)
            expected = hmac.new(
                self.secret.encode(), data.encode(), hashlib.sha256
            ).hexdigest()
            if not hmac.compare_digest(sig, expected):
                return None
            payload = json.loads(data)
            if payload.get("exp", 0) < time.time():
                return None
            return self._users.get(payload.get("user_id"))
        except Exception:
            return None

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path) as f:
                data = json.load(f)
            for d in data.get("users", []):
                u = User(**d)
                self._users[u.user_id] = u
            self._invite_codes = data.get("invite_codes", {})
        except (json.JSONDecodeError, OSError):
            pass

    def _save(self) -> None:
        with open(self.path, "w") as f:
            json.dump({
                "users":        [u.to_dict() for u in self._users.values()],
                "invite_codes": self._invite_codes,
            }, f, indent=2)


# ============================================================
# FastAPI middleware helpers
# ============================================================

def make_auth_dependency(auth_system: AuthSystem):
    """
    Returns a FastAPI dependency that requires a valid JWT token.

    Usage:
      require_auth = make_auth_dependency(auth)

      @app.get("/protected")
      async def route(user = Depends(require_auth)):
          return {"email": user.email}
    """
    try:
        from fastapi import HTTPException, Header
    except ImportError:
        return None

    async def require_auth(authorization: str = Header(None)):
        if not authorization:
            raise HTTPException(401, "Authorization header required")
        token = authorization.replace("Bearer ", "").strip()
        user  = auth_system.verify_token(token)
        if not user:
            raise HTTPException(401, "Invalid or expired token")
        if not user.is_active:
            raise HTTPException(403, "Account suspended")
        return user

    return require_auth


def make_admin_dependency(auth_system: AuthSystem):
    """Returns a dependency that requires admin role."""
    try:
        from fastapi import HTTPException, Header
    except ImportError:
        return None

    async def require_admin(authorization: str = Header(None)):
        if not authorization:
            raise HTTPException(401, "Authorization required")
        token = authorization.replace("Bearer ", "").strip()
        user  = auth_system.verify_token(token)
        if not user or user.role != "admin":
            raise HTTPException(403, "Admin access required")
        return user

    return require_admin
