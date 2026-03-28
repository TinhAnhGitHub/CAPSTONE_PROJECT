export function ensureSessionId(sessions, session, setSession) {
    if (session && sessions.some(g => g._id === session)) {
        return { status: "ok", group_id: session };
    }

    if (sessions.length > 0) {
        setSession(sessions[0]._id);
        return { status: "switched", group_id: sessions[0]._id };
    }

    return { status: "create" };
}
