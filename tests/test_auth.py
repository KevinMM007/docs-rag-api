from __future__ import annotations

from fastapi.testclient import TestClient


def _register(client: TestClient, email: str, password: str = "secret-pw-12") -> dict:
    response = client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": password},
    )
    assert response.status_code == 201, response.text
    return response.json()


def _login(client: TestClient, email: str, password: str = "secret-pw-12") -> str:
    response = client.post(
        "/api/v1/auth/login",
        data={"username": email, "password": password},
    )
    assert response.status_code == 200, response.text
    return response.json()["access_token"]


def test_register_returns_created_user(client: TestClient) -> None:
    body = _register(client, "alice@example.com")
    assert body["email"] == "alice@example.com"
    assert body["is_active"] is True
    assert isinstance(body["id"], int)
    assert "hashed_password" not in body  # never leak the hash


def test_register_duplicate_email_returns_409(client: TestClient) -> None:
    _register(client, "bob@example.com")
    response = client.post(
        "/api/v1/auth/register",
        json={"email": "bob@example.com", "password": "another-pw-99"},
    )
    assert response.status_code == 409


def test_register_rejects_short_password(client: TestClient) -> None:
    response = client.post(
        "/api/v1/auth/register",
        json={"email": "carol@example.com", "password": "short"},
    )
    assert response.status_code == 422


def test_login_returns_bearer_token(client: TestClient) -> None:
    _register(client, "dave@example.com")
    token = _login(client, "dave@example.com")
    assert token  # non-empty JWT


def test_login_wrong_password_returns_401(client: TestClient) -> None:
    _register(client, "erin@example.com")
    response = client.post(
        "/api/v1/auth/login",
        data={"username": "erin@example.com", "password": "wrong-pw-99"},
    )
    assert response.status_code == 401


def test_login_unknown_user_returns_401(client: TestClient) -> None:
    response = client.post(
        "/api/v1/auth/login",
        data={"username": "ghost@example.com", "password": "whatever-99"},
    )
    assert response.status_code == 401


def test_me_returns_current_user(client: TestClient) -> None:
    _register(client, "frank@example.com")
    token = _login(client, "frank@example.com")
    response = client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    assert response.json()["email"] == "frank@example.com"


def test_me_without_token_returns_401(client: TestClient) -> None:
    response = client.get("/api/v1/auth/me")
    assert response.status_code == 401


def test_me_with_garbage_token_returns_401(client: TestClient) -> None:
    response = client.get(
        "/api/v1/auth/me",
        headers={"Authorization": "Bearer not-a-real-jwt"},
    )
    assert response.status_code == 401


def test_me_with_non_integer_subject_returns_401(client: TestClient) -> None:
    """A token whose ``sub`` claim isn't a valid user id (e.g. an email or a
    UUID from a different schema) must be rejected rather than blow up with
    a 500 from ``int()``.
    """
    from app.core.security import create_access_token

    token = create_access_token(subject="not-a-number")
    response = client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 401


def test_me_with_unknown_user_id_returns_401(client: TestClient) -> None:
    """Token validates structurally but points at a user that was deleted (or
    never existed in this DB) - the deps layer must surface that as 401."""
    from app.core.security import create_access_token

    token = create_access_token(subject="99999")
    response = client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 401
