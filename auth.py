# auth.py - Simple OTP Authentication for Sahayak

import os
import jwt
import secrets
import requests
import smtplib
import re
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple
from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pymongo import MongoClient
from dotenv import load_dotenv
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

load_dotenv()

# Security configurations
security = HTTPBearer()

# JWT Configuration
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "sahayak_default_secret_key_change_in_production")
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Database Configuration
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME")
OTP_COLLECTION = "otp_tokens"

# 2Factor.in Configuration
TWOFACTOR_API_KEY = "0c6c2161-14ad-11f0-8b17-0200cd936042"
TWOFACTOR_BASE_URL = "https://2factor.in/API/V1"

# Email Configuration
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_USER = "mehaksharmao845@gmail.com"
SMTP_PASSWORD = "gmolzkzkedskbafx"

class AuthManager:
    def __init__(self):
        try:
            self.mongo_client = MongoClient(MONGO_URI)
            self.db = self.mongo_client[MONGO_DB_NAME]
            self.otp_collection = self.db[OTP_COLLECTION]
            
            # Create index for OTP expiration
            self.otp_collection.create_index("expires_at", expireAfterSeconds=0)
            print("✅ OTP Auth Manager initialized successfully")
        except Exception as e:
            print(f"⚠️ Auth Manager initialization error: {e}")
            self.mongo_client = None
            self.db = None
            self.otp_collection = None

    def generate_otp(self, length: int = 6) -> str:
        """Generate a random OTP"""
        return ''.join([str(secrets.randbelow(10)) for _ in range(length)])

    def send_email_otp(self, email: str) -> Tuple[bool, str]:
        """Send OTP via email"""
        try:
            otp = self.generate_otp(6)
            
            msg = MIMEMultipart()
            msg['From'] = SMTP_USER
            msg['To'] = email
            msg['Subject'] = "Sahayak - Your OTP Code"

            body = f"""
            <html>
            <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
                <div style="background: linear-gradient(135deg, #007bff, #0056b3); color: white; padding: 20px; text-align: center;">
                    <h1>🎓 Sahayak</h1>
                    <p>AI-Powered Educational Platform</p>
                </div>
                <div style="padding: 30px; background-color: #f8f9fa;">
                    <h2 style="color: #333;">Your OTP Code</h2>
                    <div style="background: white; padding: 20px; border-radius: 10px; text-align: center; margin: 20px 0;">
                        <h1 style="font-size: 36px; color: #007bff; margin: 0; letter-spacing: 5px;">{otp}</h1>
                    </div>
                    <p style="color: #666; line-height: 1.6;">
                        This OTP will expire in <strong>5 minutes</strong>. 
                        Please use it to complete your login.
                    </p>
                </div>
                <div style="background: #343a40; color: white; padding: 15px; text-align: center;">
                    <p style="margin: 0;">Best regards,<br><strong>Sahayak Team</strong></p>
                </div>
            </body>
            </html>
            """
            
            msg.attach(MIMEText(body, 'html'))

            print(f"🔍 Debug - Sending email OTP {otp} to {email}")
            
            server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            text = msg.as_string()
            server.sendmail(SMTP_USER, email, text)
            server.quit()
            
            print(f"✅ Email OTP sent to {email}")
            return True, otp
            
        except Exception as e:
            print(f"❌ Email OTP error: {e}")
            return False, ""

    def send_sms_otp(self, phone: str) -> Tuple[bool, str]:
        """Send OTP via SMS using 2Factor.in API"""
        try:
            # Format phone number for API
            formatted_phone = self._format_phone_for_api(phone)
            otp = self.generate_otp(6)
            
            # Construct API URL with our OTP (not AUTOGEN)
            api_url = f"{TWOFACTOR_BASE_URL}/{TWOFACTOR_API_KEY}/SMS/{formatted_phone}/{otp}"
            
            print(f"🔍 Debug - Sending OTP {otp} to {formatted_phone}")
            print(f"🔍 Debug - API URL: {api_url}")
            
            # Make API request
            response = requests.get(api_url, timeout=10)
            
            if response.status_code == 200:
                try:
                    response_data = response.json()
                    print(f"🔍 Debug - API Response: {response_data}")
                    if response_data.get("Status") == "Success":
                        print(f"✅ SMS OTP sent to {formatted_phone}")
                        return True, otp
                    else:
                        print(f"❌ 2Factor API error: {response_data}")
                        return False, ""
                except:
                    # Sometimes API returns plain text instead of JSON
                    print(f"✅ SMS OTP sent to {formatted_phone} (non-JSON response)")
                    return True, otp
            else:
                print(f"❌ 2Factor API HTTP error: {response.status_code}")
                print(f"❌ Response: {response.text}")
                return False, ""
                
        except Exception as e:
            print(f"❌ SMS OTP error: {e}")
            return False, ""

    def _format_phone_for_api(self, phone: str) -> str:
        """Format phone number for 2Factor API"""
        phone = re.sub(r'[^\d+]', '', phone)
        if not phone.startswith('+91'):
            if phone.startswith('91'):
                phone = '+' + phone
            else:
                phone = '+91' + phone
        return phone

    def store_otp(self, identifier: str, otp: str, otp_type: str = "login") -> None:
        """Store OTP in MongoDB with expiration"""
        if not self.otp_collection:
            print("❌ Cannot store OTP - database not available")
            return
            
        expires_at = datetime.utcnow() + timedelta(minutes=5)
        
        otp_data = {
            "identifier": identifier,
            "otp": otp,
            "type": otp_type,
            "created_at": datetime.utcnow(),
            "expires_at": expires_at,
            "is_used": False
        }
        
        # Remove any existing OTP for this identifier
        self.otp_collection.delete_many({"identifier": identifier, "type": otp_type})
        
        # Store new OTP
        result = self.otp_collection.insert_one(otp_data)
        print(f"✅ OTP stored for {identifier}: {otp} (expires at {expires_at})")

    def verify_otp(self, identifier: str, otp: str, otp_type: str = "login") -> bool:
        """Verify OTP from MongoDB"""
        if not self.otp_collection:
            print("❌ Cannot verify OTP - database not available")
            return False
        
        print(f"🔍 Debug - Verifying OTP for: {identifier}, OTP: {otp}, Type: {otp_type}")
        
        # First, let's see what OTPs exist for this identifier
        all_otps = list(self.otp_collection.find({"identifier": identifier}))
        print(f"🔍 Debug - Found {len(all_otps)} OTPs for identifier {identifier}")
        
        for otp_record in all_otps:
            print(f"🔍 Debug - OTP Record: {otp_record['otp']}, Used: {otp_record['is_used']}, Expires: {otp_record['expires_at']}")
            
        otp_record = self.otp_collection.find_one({
            "identifier": identifier,
            "otp": otp,
            "type": otp_type,
            "is_used": False,
            "expires_at": {"$gt": datetime.utcnow()}
        })

        print(f"🔍 Debug - Query result: {otp_record}")

        if otp_record:
            # Mark OTP as used
            self.otp_collection.update_one(
                {"_id": otp_record["_id"]},
                {"$set": {"is_used": True}}
            )
            print(f"✅ OTP verified for {identifier}")
            return True
        
        print(f"❌ OTP verification failed for {identifier}")
        return False

    def create_access_token(self, data: Dict[str, Any]) -> str:
        """Create a JWT access token"""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        to_encode.update({"exp": expire, "type": "access"})
        return jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)

    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify and decode a JWT token"""
        try:
            payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired"
            )
        except jwt.JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials"
            )

    def get_current_user(self, credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
        """Get current authenticated user from JWT token"""
        try:
            token = credentials.credentials
            payload = self.verify_token(token)
            
            user_id = payload.get("sub")
            identifier = payload.get("identifier")
            method = payload.get("method")
            
            if not user_id:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token"
                )

            return {
                "_id": user_id, 
                "identifier": identifier,
                "method": method,
                "role": "student"
            }
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Authentication failed: {str(e)}"
            )

# Initialize auth manager
try:
    auth_manager = AuthManager()
except Exception as e:
    print(f"⚠️ Could not initialize auth manager: {e}")
    auth_manager = None

# Dependency functions for FastAPI
def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current user dependency"""
    if auth_manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication service not available"
        )
    return auth_manager.get_current_user(credentials)

def get_current_active_user(current_user: dict = Depends(get_current_user)):
    """Get current active user dependency"""
    return current_user