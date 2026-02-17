[SwapRecipe] Starting swap for user: 7e31e7c4-add8-4433-ae70-f978f0df4f35
[SwapRecipe] Recipe to replace: d275e741-8e75-4623-821e-190d408571b5
[SwapRecipe] Swap context: I want a sweeter dish please
[SwapRecipe] Target week: 1
[SwapRecipe] Current plan has 3 recipes
[SwapRecipe] Swapping recipe: Sautéed Chicken with Vegetables
[SwapRecipe] Added current plan recipe 05fe0fdd-1711-4c1f-9e6c-f5f1ff43960e to exclusions
[SwapRecipe] Added current plan recipe b62d3193-d8c8-4dca-8f91-2e183ce155dd to exclusions
[SwapRecipe] Added recipe to swap d275e741-8e75-4623-821e-190d408571b5 to exclusions
[SwapRecipe] Total exclusions: 3 recipes
================================================================================
[SwapRecipe: RUNTIME CONTEXT INITIALIZED]
user_id: 7e31e7c4-add8-4433-ae70-f978f0df4f35
frequency: 3
cuisine: Chinese
dietary_restrictions: []
allergens: []
current recipes: 3
================================================================================
[SwapRecipe] Invoking agent for swap...

[call_agent_reasoner] === ENTERED ===
[call_agent_reasoner] Current candidate_recipes: ['05fe0fdd-1711-4c1f-9e6c-f5f1ff43960e', 'd275e741-8e75-4623-821e-190d408571b5', 'b62d3193-d8c8-4dca-8f91-2e183ce155dd']
[call_agent_reasoner] Message count BEFORE trim: 1
[call_agent_reasoner] Message count AFTER trim: 1
[call_agent_reasoner] Invoking agent...
[call_agent_reasoner] Agent response type: <class 'langchain_core.messages.ai.AIMessage'>
[call_agent_reasoner] Agent response content: No content
[call_agent_reasoner] Has tool_calls: True
[call_agent_reasoner] Tool calls: [{'name': 'get_recipe_candidates', 'args': {'intent_query': 'Chinese sweet recipe for intermediate'}, 'id': 'call_urNWOECYj8mxXXkSNkf2vzdk', 'type': 'tool_call'}]

[route_agent_action] === ROUTING ===
[route_agent_action] Last message has tool_calls: True
[route_agent_action] Current candidate_recipes: ['05fe0fdd-1711-4c1f-9e6c-f5f1ff43960e', 'd275e741-8e75-4623-821e-190d408571b5', 'b62d3193-d8c8-4dca-8f91-2e183ce155dd']
[route_agent_action] → Routing to 'tool'

[execute_tool] === ENTERED ===
[execute_tool] Current candidate_recipes BEFORE tool: ['05fe0fdd-1711-4c1f-9e6c-f5f1ff43960e', 'd275e741-8e75-4623-821e-190d408571b5', 'b62d3193-d8c8-4dca-8f91-2e183ce155dd']
[execute_tool] Tool name: get_recipe_candidates
[execute_tool] Tool args: {'intent_query': 'Chinese sweet recipe for intermediate'}

[TOOL: get_recipe_candidates] === CALLED ===
[TOOL] intent_query: Chinese sweet recipe for intermediate
[TOOL] ✓ Using runtime context
[TOOL] user_id: 7e31e7c4-add8-4433-ae70-f978f0df4f35
[TOOL] exclude_ids: 3 recipes
[TOOL] frequency: 3
[TOOL] cuisine: Chinese
[TOOL] skill_level: intermediate
[get_recipe_candidates_hybrid] Excluding 3 recipes: ['05fe0fdd-1711-4c1f-9e6c-f5f1ff43960e', 'b62d3193-d8c8-4dca-8f91-2e183ce155dd', 'd275e741-8e75-4623-821e-190d408571b5']...
[get_recipe_candidates_hybrid] User preferences - Cuisine: Chinese, Skill: intermediate
[TOOL: get_recipe_candidates] 🔄 SWAP MODE: Stored 3 candidates in swap_candidates
[TOOL: get_recipe_candidates] Current candidate_recipes: ['05fe0fdd-1711-4c1f-9e6c-f5f1ff43960e', 'd275e741-8e75-4623-821e-190d408571b5', 'b62d3193-d8c8-4dca-8f91-2e183ce155dd']
[TOOL: get_recipe_candidates] Swap candidates: ['1ca324e1-81d6-45ee-9bcf-2ad95a697aea', '53b717bc-734e-43a0-8b6d-3fbf949e0bc3', 'aa169899-2bc9-4022-827c-9c31473e57f1']...
[TOOL: get_recipe_candidates] ✅ Found 3 recipes
[TOOL: get_recipe_candidates] Similarity scores: ['0.555', '0.505', '0.500']
[TOOL: get_recipe_candidates] Returning:
--- Recipe Candidates ---

1. ID: 1ca324e1-81d6-45ee-9bcf-2ad95a697aea, Name: Sweet and Sour Pork, Difficulty: medium, Score: 0.555
2. ID: 53b717bc-734e-43a0-8b6d-3fbf949e0bc3, Name: Chinese Orange Ch...
   [execute_tool] Tool result type: <class 'dict'>
   [execute_tool] Tool result: {'messages': [ToolMessage(content='\n--- Recipe Candidates ---\n1. ID: 1ca324e1-81d6-45ee-9bcf-2ad95a697aea, Name: Sweet and Sour Pork, Difficulty: medium, Score: 0.555\n2. ID: 53b717bc-734e-43a0-8b6d-3fbf949e0bc3, Name: Chinese Orange Chicken, Difficulty: medium, Score: 0.505\n3. ID: aa169899-2bc9-4022-827c-9c31473e57f1, Name: General Tsos Chicken, Difficulty: medium, Score: 0.500\n\n🔄 SWAP MODE - CHOOSE THE BEST:\n- #1 has the highest relevance score (0.555)\n- Review all 3 options above\n- Respond with ONLY the UUID of the best match\n- Example: Just respond with the ID, nothing else\n- The backend will handle the rest\n\n✓ Found all 3 requested recipes.\n', name='get_recipe_candidates', tool_call_id='call_urNWOECYj8mxXXkSNkf2vzdk')]}
   [execute_tool] ✓ Syncing runtime state → graph state
   [execute_tool] candidate_recipes: 3 recipes

[call_agent_reasoner] === ENTERED ===
[call_agent_reasoner] Current candidate_recipes: ['05fe0fdd-1711-4c1f-9e6c-f5f1ff43960e', 'd275e741-8e75-4623-821e-190d408571b5', 'b62d3193-d8c8-4dca-8f91-2e183ce155dd']
[call_agent_reasoner] Message count BEFORE trim: 3
[call_agent_reasoner] Message count AFTER trim: 3
[call_agent_reasoner] Invoking agent...
[call_agent_reasoner] Agent response type: <class 'langchain_core.messages.ai.AIMessage'>
[call_agent_reasoner] Agent response content: 1ca324e1-81d6-45ee-9bcf-2ad95a697aea
[call_agent_reasoner] Has tool_calls: False
[call_agent_reasoner] No tool calls, ending cycle

[route_agent_action] === ROUTING ===
[route_agent_action] Last message has tool_calls: False
[route_agent_action] Current candidate_recipes: ['05fe0fdd-1711-4c1f-9e6c-f5f1ff43960e', 'd275e741-8e75-4623-821e-190d408571b5', 'b62d3193-d8c8-4dca-8f91-2e183ce155dd']
[route_agent_action] → Routing to 'end' (finalizer)

[finalize_plan_output] === ENTERED ===
[finalize_plan_output] Full state keys: dict_keys(['messages', 'user_id', 'user_goal', 'frequency', 'exclude_ids', 'candidate_recipes', 'next_action'])
[finalize_plan_output] State: {'messages': [HumanMessage(content='RECIPE SWAP REQUEST:\n\nRecipe to replace: Sautéed Chicken with Vegetables (ID: d275e741-8e75-4623-821e-190d408571b5)\nUser\'s reason: I want a sweeter dish please\n\nYOUR JOB:\n1. Search for ONE replacement recipe using get_recipe_candidates\n2. Review the search results with recipe names, difficulty, and relevance scores\n3. Pick the BEST recipe ID from the results (consider: relevance score, cuisine, difficulty)\n4. Respond with ONLY that recipe ID (nothing else)\n\nExample response: "8a9cf1e4-1e3f-49e7-babe-01941de08134"\n\nThe backend will handle inserting it into the meal plan.', additional_kwargs={}, response_metadata={}), AIMessage(content='', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 20, 'prompt_tokens': 1572, 'total_tokens': 1592, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_provider': 'openai', 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_373a14eb6f', 'id': 'chatcmpl-DA3X3dJCBjr5gHqeeGGhAQSJdyicM', 'service_tier': 'default', 'finish_reason': 'tool_calls', 'logprobs': None}, id='lc_run--eb66353e-d558-4180-bf1b-acd43da96744-0', tool_calls=[{'name': 'get_recipe_candidates', 'args': {'intent_query': 'Chinese sweet recipe for intermediate'}, 'id': 'call_urNWOECYj8mxXXkSNkf2vzdk', 'type': 'tool_call'}], usage_metadata={'input_tokens': 1572, 'output_tokens': 20, 'total_tokens': 1592, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}}), ToolMessage(content='\n--- Recipe Candidates ---\n1. ID: 1ca324e1-81d6-45ee-9bcf-2ad95a697aea, Name: Sweet and Sour Pork, Difficulty: medium, Score: 0.555\n2. ID: 53b717bc-734e-43a0-8b6d-3fbf949e0bc3, Name: Chinese Orange Chicken, Difficulty: medium, Score: 0.505\n3. ID: aa169899-2bc9-4022-827c-9c31473e57f1, Name: General Tsos Chicken, Difficulty: medium, Score: 0.500\n\n🔄 SWAP MODE - CHOOSE THE BEST:\n- #1 has the highest relevance score (0.555)\n- Review all 3 options above\n- Respond with ONLY the UUID of the best match\n- Example: Just respond with the ID, nothing else\n- The backend will handle the rest\n\n✓ Found all 3 requested recipes.\n', name='get_recipe_candidates', tool_call_id='call_urNWOECYj8mxXXkSNkf2vzdk'), AIMessage(content='1ca324e1-81d6-45ee-9bcf-2ad95a697aea', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 24, 'prompt_tokens': 1817, 'total_tokens': 1841, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 1536}}, 'model_provider': 'openai', 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_373a14eb6f', 'id': 'chatcmpl-DA3X5dBr8c7ABb9Xlm0GEwy4mxwEd', 'service_tier': 'default', 'finish_reason': 'stop', 'logprobs': None}, id='lc_run--7d924dbc-08e1-4123-a065-07f0c3f6f338-0', usage_metadata={'input_tokens': 1817, 'output_tokens': 24, 'total_tokens': 1841, 'input_token_details': {'audio': 0, 'cache_read': 1536}, 'output_token_details': {'audio': 0, 'reasoning': 0}})], 'user_id': '7e31e7c4-add8-4433-ae70-f978f0df4f35', 'user_goal': 'health', 'frequency': 3, 'exclude_ids': ['05fe0fdd-1711-4c1f-9e6c-f5f1ff43960e', 'b62d3193-d8c8-4dca-8f91-2e183ce155dd', 'd275e741-8e75-4623-821e-190d408571b5'], 'candidate_recipes': ['05fe0fdd-1711-4c1f-9e6c-f5f1ff43960e', 'd275e741-8e75-4623-821e-190d408571b5', 'b62d3193-d8c8-4dca-8f91-2e183ce155dd'], 'next_action': 'end'}
[finalize_plan_output] Final selections: ['05fe0fdd-1711-4c1f-9e6c-f5f1ff43960e', 'd275e741-8e75-4623-821e-190d408571b5', 'b62d3193-d8c8-4dca-8f91-2e183ce155dd']
[finalize_plan_output] Final selections type: <class 'list'>
[finalize_plan_output] Final selections length: 3
[finalize_plan_output] ✅ Returning 3 recipes
[SwapRecipe] Agent response: 1ca324e1-81d6-45ee-9bcf-2ad95a697aea
[SwapRecipe] Extracted recipe ID: 1ca324e1-81d6-45ee-9bcf-2ad95a697aea
[SwapRecipe] Swapped d275e741-8e75-4623-821e-190d408571b5 → 1ca324e1-81d6-45ee-9bcf-2ad95a697aea
[SwapRecipe] Updated schedule: [{"recipe_id": "05fe0fdd-1711-4c1f-9e6c-f5f1ff43960e", "order": 0}, {"recipe_id": "1ca324e1-81d6-45ee-9bcf-2ad95a697aea", "order": 1}, {"recipe_id": "b62d3193-d8c8-4dca-8f91-2e183ce155dd", "order": 2}]
[SwapRecipe] ✅ Database updated successfully
[SwapRecipe] Saved to Supabase: [{"recipe_id": "05fe0fdd-1711-4c1f-9e6c-f5f1ff43960e", "order": 0}, {"recipe_id": "1ca324e1-81d6-45ee-9bcf-2ad95a697aea", "order": 1}, {"recipe_id": "b62d3193-d8c8-4dca-8f91-2e183ce155dd", "order": 2}]
[SwapRecipe] ✅ Swap completed: Sautéed Chicken with Vegetables → Sweet and Sour Pork
[SwapCleanup] Deleted progress for swapped recipe d275e741-8e75-4623-821e-190d408571b5 in week 1
[SwapRecipe: RUNTIME CONTEXT CLEARED]
