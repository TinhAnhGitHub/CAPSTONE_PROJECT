export function ensureGroupId(groups, currentGroup, setCurrentGroup) {
    if (currentGroup) return { status: "ok", group_id: currentGroup };

    if (groups.length > 0) {
        setCurrentGroup(groups[0]._id);
        return { status: "switched", group_id: groups[0]._id };
    }

    return { status: "create" };
    // if (currentGroup && groups.some(g => g._id === currentGroup)) {
    //     return { status: "ok", group_id: currentGroup };
    // }

    // if (groups.length > 0) {
    //     setCurrentGroup(groups[0]._id);
    //     return { status: "switched", group_id: groups[0]._id };
    // }

    // return { status: "create" };
}

