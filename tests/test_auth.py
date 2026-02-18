def test_signup(client):
    """Test user signup"""
    response = client.post(
        "/auth/register",
        json={
            "email": "newuser@example.com",
            "password": "new123",
            "first_name": "John",
            "last_name": "Doe",
            "cuisine": "Mexican",
            "frequency": 4,
            "skill_level": "beginner",
            "user_goal": "Eat Healthier",
        },
    )
    assert response.status_code == 201
    data = response.json()
    assert data["success"] is True
    assert "access_token" in data
    assert data["user"]["email"] == "newuser@example.com"


def test_login(client, test_user):
    """Test user login"""
    response = client.post(
        "/auth/login",
        json={"email": "testuser@example.com", "password": "test123"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["user"]["email"] == "testuser@example.com"


def test_login_invalid_password(client, test_user):
    """Test login with wrong password"""
    response = client.post(
        "/auth/login",
        json={"email": "testuser@example.com", "password": "wrongpassword"},
    )
    assert response.status_code == 401
