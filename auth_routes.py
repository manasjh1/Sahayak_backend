# auth_routes.py

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Literal
from datetime import datetime
import re

try:
    from auth import auth_manager
    AUTH_AVAILABLE = True
except ImportError:
    AUTH_AVAILABLE = False
    auth_manager = None

auth_router = APIRouter(prefix="/auth", tags=["Authentication"])

class SendOTPRequest(BaseModel):
    identifier: str
    method: Literal["email", "sms"]

class VerifyOTPRequest(BaseModel):
    identifier: str
    otp: str
    method: Literal["email", "sms"]

@auth_router.post("/send-otp")
async def send_otp(request: SendOTPRequest):
    # ... (This endpoint can remain the same)
    # It's responsible for sending the OTP, which is still a valid function
    # for a simple login flow.
    try:
        # ... (same validation and sending logic as before)
        if not AUTH_AVAILABLE or not auth_manager:
            raise HTTPException(status_code=503, detail="Auth service unavailable")

        if request.method == "email":
            success, otp = auth_manager.send_email_otp(request.identifier)
        elif request.method == "sms":
            success, otp = auth_manager.send_sms_otp(request.identifier)

        if success:
            auth_manager.store_otp(request.identifier, otp, "login")
            return JSONResponse(content={"message": f"OTP sent successfully to your {request.method}"})
        else:
            raise HTTPException(status_code=500, detail=f"Failed to send {request.method} OTP")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@auth_router.post("/verify-otp")
async def verify_otp(request: VerifyOTPRequest):
    """Verify OTP and confirm login (no JWT returned)"""
    try:
        if not AUTH_AVAILABLE or not auth_manager:
            raise HTTPException(status_code=503, detail="Auth service unavailable")

        if auth_manager.verify_otp(request.identifier, request.otp, "login"):
            # Instead of a JWT, return a success message
            return JSONResponse(content={
                "message": "OTP verified successfully. You are now logged in.",
                "user_id": request.identifier, # You can return the user identifier
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
        raise HTTPException(status_code=500, detail=str(e))