import clsx from "clsx";
import { useStore } from "@/stores/chat";
import GroupDropdownList from "./GroupDropdownList";
import useEdit from "@/api/services/hooks/edit";

export default function Group({ group, onEdit }) {
  const currentGroup = useStore((state) => state.currentGroup);
  const setCurrentGroup = useStore((state) => state.setCurrentGroup);

  const {
    isEditing,
    editValue,
    setEditValue,
    startEditing,
    saveEdit,
    cancelEdit,
  } = useEdit({
    initialValue: group.name,
    onSave: (value) => onEdit?.(group._id, value),
  });

  const handleSelectGroup = () => {
    if (!isEditing) {
      setCurrentGroup(group._id);
    }
  };

  return isEditing ? (
    <input
      className="relative mx-1 my-1 py-2 px-3 bg-surface rounded-lg text-sm text-text outline-none focus:ring-2 focus:ring-accent/50"
      autoFocus
      value={editValue}
      onChange={(e) => setEditValue(e.target.value)}
      onBlur={saveEdit}
      onKeyDown={(e) => {
        if (e.key === "Enter") saveEdit();
        if (e.key === "Escape") cancelEdit();
      }}
    />
  ) : (
    <div
      className={clsx(
        "relative mx-1 my-0.5 py-2 px-3 rounded-lg cursor-pointer transition-colors",
        "text-text-muted hover:text-text hover:bg-white/5",
        currentGroup === group._id && "bg-white/10 text-text",
        "group"
      )}
      onClick={handleSelectGroup}
    >
      <div className="text-sm truncate pr-6">{group.name}</div>

      <div className="absolute right-2 top-1/2 -translate-y-1/2 rounded-md p-1 hover:bg-white/10 cursor-pointer block md:hidden md:group-hover:block has-data-open:block">
        <GroupDropdownList group={group} onStartEdit={startEditing} />
      </div>
    </div>
  );
}
