"""
Demo router — what a logged-out visitor is allowed to do right now.

The frontend calls this on load to decide what to render: how many free queries
are left, and whether live research is currently paused by the spend ceiling.
It also mints the visitor cookie, which is why cookie-setting lives here rather
than on ``/api/research`` — attaching a ``Set-Cookie`` to a streaming response
works, but a plain JSON endpoint is a far easier place to get it right and to
test.
"""

import logging

from fastapi import APIRouter, Request, Response

from app.config import get_settings
from app.routers.auth import Principal, get_principal
from app.services import budget
from app.services.anonymous import (
    new_anon_id,
    peek_quota,
    read_anon_id,
    set_anon_cookie,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/demo", tags=["demo"])


@router.get("/status")
async def demo_status(request: Request, response: Response) -> dict:
    """
    The visitor's current standing: allowance left, and whether live runs work.

    Deliberately readable by signed-in users too — they get ``anonymous: false``
    and no quota, but the same ``live`` block, so one call tells the UI whether
    to show the degraded banner regardless of who is asking.
    """
    settings = get_settings()

    if not settings.anonymous_demo_enabled:
        return {
            "enabled": False,
            "anonymous": True,
            "quota": None,
            "live": {"available": False, "reason": "disabled", "message": ""},
        }

    principal: Principal = await get_principal(request)

    quota = None
    if principal.is_anonymous:
        anon_id = read_anon_id(request)
        if not anon_id:
            anon_id = new_anon_id()
            set_anon_cookie(response, anon_id)
        quota = (await peek_quota(request, anon_id)).as_dict()

    status = await budget.check_budget(anonymous=principal.is_anonymous)

    return {
        "enabled": True,
        "anonymous": principal.is_anonymous,
        "quota": quota,
        "live": {
            "available": status.allowed,
            "reason": status.reason,
            "message": status.message,
        },
    }
