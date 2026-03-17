def decide_action(emotion, intensity, stress, energy, time_of_day):
    
    action = "pause"
    when = "later_today"

    # High stress situations
    if stress >= 7:
        if intensity >= 4:
            action = "box_breathing"
            when = "now"
        else:
            action = "grounding"
            when = "within_15_min"

    # Low energy cases
    elif energy <= 3:
        if time_of_day in ["night", "late_night"]:
            action = "rest"
            when = "tonight"
        else:
            action = "rest"
            when = "now"

    # Calm / focused state
    elif emotion in ["calm", "focused"]:
        if energy >= 7:
            action = "deep_work"
            when = "within_15_min"
        else:
            action = "light_planning"
            when = "later_today"

    # Mixed emotion
    elif emotion == "mixed":
        action = "journaling"
        when = "within_15_min"

    # Neutral state
    elif emotion == "neutral":
        if energy >= 6:
            action = "movement"
            when = "within_15_min"
        else:
            action = "light_planning"
            when = "later_today"

    # Restless / overwhelmed
    elif emotion in ["restless", "overwhelmed"]:
        action = "grounding"
        when = "now"

    return action, when