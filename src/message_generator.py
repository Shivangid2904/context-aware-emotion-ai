def generate_message(state, intensity, action):
    
    if state == "overwhelmed":
        return "You seem quite overwhelmed right now. Let's slow things down. Try a short grounding or breathing exercise."

    if state == "restless":
        return "It looks like your mind is a bit restless. A quick grounding exercise might help bring your focus back."

    if state == "calm":
        return "You seem calm and balanced. This might be a great time to focus on something meaningful."

    if state == "focused":
        return "You're in a focused state. Consider starting deep work while your energy is aligned."

    if state == "neutral":
        return "You're in a stable state. A light planning session could help organize your next steps."

    return "Take a moment to pause and check in with yourself."