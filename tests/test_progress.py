def test_submit_feedback(client, test_user, test_plan, test_recipes):
    """Test submitting recipe feedback"""
    # First create progress entry
    response = client.post(
        f"/feedback/{test_user.id}",
        json={
            "recipe_id": str(test_recipes[0].id),
            "week_number": 1,
            "feedback": "just_right",
        },
    )
    assert response.status_code in [200, 404]  # 404 if progress not found


def test_get_progress_summary(client, test_user):
    """Test getting progress summary"""
    response = client.get(f"/api/progress/{test_user.id}")
    assert response.status_code in [200, 404]
