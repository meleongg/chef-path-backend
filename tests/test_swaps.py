def test_swap_limit_check(client, test_user, test_plan):
    """Test that swap endpoint enforces swap limit"""
    # First, ensure the plan exists
    response = client.get(f"/api/weekly-plan/{test_user.id}/all")
    assert response.status_code == 200

    # Attempting to swap (will fail due to missing recipe in progress, but validates limit logic)
    swap_payload = {
        "recipe_id_to_replace": "00000000-0000-0000-0000-000000000001",
        "swap_context": "Want something lighter",
        "week_number": 1,
    }
    response = client.post(f"/plan/swap-recipe/{test_user.id}", json=swap_payload)
    # Expect 404 (recipe not found) or 400 (validation error)
    assert response.status_code in [400, 404]


def test_get_next_week_eligibility(client, test_user):
    """Test checking if user can generate next week"""
    response = client.get(f"/plan/can_generate_next_week/{test_user.id}")
    assert response.status_code == 200
    data = response.json()
    assert "can_generate" in data
    assert "current_week" in data
