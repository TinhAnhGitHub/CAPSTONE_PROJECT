import { useState } from "react";

export default function useEdit({ initialValue, onSave }) {
    const [isEditing, setIsEditing] = useState(false);
    const [editValue, setEditValue] = useState(initialValue);

    const startEditing = () => {
        setEditValue(initialValue);
        setIsEditing(true);
    };

    const saveEdit = () => {
        if (editValue.trim()) {
            onSave(editValue.trim());
        }
        setIsEditing(false);
    };

    const cancelEdit = () => {
        setEditValue(initialValue);
        setIsEditing(false);
    };

    return {
        isEditing,
        editValue,
        setEditValue,
        startEditing,
        saveEdit,
        cancelEdit,
    };
}
