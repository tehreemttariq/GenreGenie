"""Authentication: signup and login"""
try:
    import bcrypt
    HAS_BCRYPT = True
except ImportError:
    HAS_BCRYPT = False

import hashlib
import secrets
from db import add_user, get_user_by_username

def hash_password(password):
    """Hash password with bcrypt or PBKDF2"""
    if HAS_BCRYPT:
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode(), salt).decode()
    else:
        # Fallback to PBKDF2
        salt = secrets.token_hex(16)
        pwd_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
        return f"{salt}${pwd_hash.hex()}"

def verify_password(password, password_hash):
    """Verify password against hash"""
    try:
        if HAS_BCRYPT:
            return bcrypt.checkpw(password.encode(), password_hash.encode())
        else:
            # PBKDF2 fallback
            salt, _ = password_hash.split('$')
            return hash_password(password, salt) == password_hash
    except:
        return False

def signup(username, password, confirm_password):
    """Create new user account"""
    # Validation
    if not username or len(username) < 3:
        return False, "Username must be at least 3 characters"
    
    if not password or len(password) < 6:
        return False, "Password must be at least 6 characters"
    
    if password != confirm_password:
        return False, "Passwords don't match"
    
    # Check if user exists
    if get_user_by_username(username):
        return False, "Username already exists"
    
    # Hash and save
    pwd_hash = hash_password(password)
    user_id = add_user(username, pwd_hash)
    
    if user_id:
        return True, user_id
    else:
        return False, "Error creating account"

def login(username, password):
    """Authenticate user"""
    result = get_user_by_username(username)
    
    if not result:
        return False, "User not found"
    
    user_id, pwd_hash = result
    
    if not verify_password(password, pwd_hash):
        return False, "Incorrect password"
    
    return True, user_id
