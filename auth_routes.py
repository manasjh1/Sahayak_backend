# auth_routes.py - Simple OTP Authentication for Sahayak

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Literal
from datetime import datetime
import re

# Import from auth module
try:
    from auth import auth_manager
    AUTH_AVAILABLE = True
except ImportError:
    print("⚠️ Auth module not available")
    AUTH_AVAILABLE = False
    auth_manager = None

# Create router
auth_router = APIRouter(prefix="/auth", tags=["Authentication"])

# Pydantic models
class SendOTPRequest(BaseModel):
    identifier: str  # email or phone
    method: Literal["email", "sms"]  # email or sms

class VerifyOTPRequest(BaseModel):
    identifier: str  # email or phone
    otp: str
    method: Literal["email", "sms"]  # email or sms

@auth_router.post("/send-otp")
async def send_otp(request: SendOTPRequest):
    """Send OTP to email or phone"""
    try:
        if not AUTH_AVAILABLE or not auth_manager:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Authentication service not available"
            )

        # Validate identifier format
        if request.method == "email":
            if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', request.identifier):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid email format"
                )
        elif request.method == "sms":
            # Format phone number for Indian format
            phone = re.sub(r'[^\d+]', '', request.identifier)
            if not re.match(r'^(\+91)?[6-9]\d{9}$', phone):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid phone number format (Indian format required)"
                )

        # Send OTP based on method
        if request.method == "email":
            success, otp = auth_manager.send_email_otp(request.identifier)
            if success:
                auth_manager.store_otp(request.identifier, otp, "login")
                message = "OTP sent successfully to your email"
            else:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to send email OTP"
                )
        
        elif request.method == "sms":
            success, otp = auth_manager.send_sms_otp(request.identifier)
            if success:
                auth_manager.store_otp(request.identifier, otp, "login")
                message = "OTP sent successfully to your phone"
            else:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to send SMS OTP"
                )

        return JSONResponse(content={
            "message": message,
            "identifier": request.identifier,
            "method": request.method,
            "timestamp": datetime.utcnow().isoformat()
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"OTP sending failed: {str(e)}"
        )

@auth_router.post("/verify-otp")
async def verify_otp(request: VerifyOTPRequest):
    """Verify OTP and return JWT token"""
    try:
        if not AUTH_AVAILABLE or not auth_manager:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Authentication service not available"
            )

        # Verify OTP
        if auth_manager.verify_otp(request.identifier, request.otp, "login"):
            
            # Generate JWT token
            access_token = auth_manager.create_access_token({
                "sub": f"user_{request.identifier}",
                "identifier": request.identifier,
                "method": request.method
            })

            return JSONResponse(content={
                "message": "OTP verified successfully",
                "access_token": access_token,
                "token_type": "bearer",
                "identifier": request.identifier,
                "method": request.method,
                "timestamp": datetime.utcnow().isoformat()
            })
        
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid or expired OTP"
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"OTP verification failed: {str(e)}"
        )