"""Password hashing, signed bearer tokens, and tenant membership authorization."""
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import os
import secrets
from jose import JWTError, jwt
from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.orm import Session
from backend.database.models import Membership, Tenant, User
from backend.database.session import get_db

ALGORITHM = "HS256"
TOKEN_TTL_MINUTES = int(os.getenv("AURA_TOKEN_TTL_MINUTES", "60"))
# If unset, a fresh secret is generated for this process; set AURA_SECRET_KEY to
# retain login sessions across restarts. Never commit that value.
JWT_SECRET = os.getenv("AURA_SECRET_KEY") or secrets.token_urlsafe(48)
bearer = HTTPBearer(auto_error=False)


@dataclass
class AuthContext:
    user: User
    tenant: Tenant
    roles: set[str]


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=2**14, r=8, p=1, dklen=32)
    return "scrypt$16384$" + salt.hex() + "$" + digest.hex()


def verify_password(password: str, encoded: str) -> bool:
    try:
        algorithm, cost, salt_hex, digest_hex = encoded.split("$", 3)
        if algorithm != "scrypt" or int(cost) != 16384: return False
        actual = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt_hex), n=int(cost), r=8, p=1, dklen=32)
        return hmac.compare_digest(actual.hex(), digest_hex)
    except (ValueError, TypeError): return False


def issue_token(user: User, tenant: Tenant, roles: set[str]) -> str:
    now = datetime.now(timezone.utc)
    claims = {"sub": user.id, "tid": tenant.id, "roles": sorted(roles), "ver": user.auth_version,
              "iat": now, "exp": now + timedelta(minutes=TOKEN_TTL_MINUTES)}
    return jwt.encode(claims, JWT_SECRET, algorithm=ALGORITHM)


def token_ttl_seconds() -> int:
    return TOKEN_TTL_MINUTES * 60


def current_context(request: Request, credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
                    db: Session = Depends(get_db)) -> AuthContext:
    unauthorized = HTTPException(status_code=401, detail="Authentication required", headers={"WWW-Authenticate": "Bearer"})
    if credentials is None or credentials.scheme.lower() != "bearer": raise unauthorized
    try: claims = jwt.decode(credentials.credentials, JWT_SECRET, algorithms=[ALGORITHM])
    except JWTError: raise unauthorized
    user = db.get(User, claims.get("sub"))
    tenant = db.get(Tenant, claims.get("tid"))
    if user is None or tenant is None or not user.is_active or claims.get("ver") != user.auth_version: raise unauthorized
    membership = db.scalar(select(Membership).where(Membership.tenant_id == tenant.id, Membership.user_id == user.id))
    if membership is None: raise unauthorized
    roles = {role.strip().upper() for role in membership.role.replace(";", ",").split(",") if role.strip()}
    request.state.tenant_id = tenant.id
    request.state.user_id = user.id
    return AuthContext(user=user, tenant=tenant, roles=roles)


def require_roles(*allowed: str):
    required = {role.upper() for role in allowed}
    def dependency(context: AuthContext = Depends(current_context)):
        if not context.roles.intersection(required) and not context.roles.intersection({"OWNER", "ADMIN"}):
            raise HTTPException(status_code=403, detail="Insufficient role for this operation")
        return context
    return dependency
