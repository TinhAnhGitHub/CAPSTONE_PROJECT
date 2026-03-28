import { XMarkIcon } from "@heroicons/react/20/solid";

/**
 * @param {Object} props
 * @param {string} props.label - Text to display
 * @param {React.ReactNode} [props.icon] - Icon component to display on the left
 * @param {string} [props.image] - Image URL to display on the left (alternative to icon)
 * @param {() => void} [props.onDelete] - Callback when delete button is clicked
 * @param {"sm" | "md" | "lg"} [props.size] - Size variant
 * @param {"default" | "primary" | "success" | "warning" | "error"} [props.variant] - Color variant
 * @param {boolean} [props.disabled] - Disable interactions
 * @param {string} [props.className] - Additional CSS classes
 */
export default function Chip({
    label,
    icon,
    image,
    onDelete,
    size = "md",
    variant = "default",
    disabled = false,
    className = "",
}) {
    const sizeClasses = {
        sm: "text-xs px-2 py-0.5 gap-1",
        md: "text-sm px-3 py-1 gap-1.5",
        lg: "text-base px-4 py-1.5 gap-2",
    };

    const iconSizeClasses = {
        sm: "h-3 w-3",
        md: "h-4 w-4",
        lg: "h-5 w-5",
    };

    const imageSizeClasses = {
        sm: "h-4 w-4",
        md: "h-5 w-5",
        lg: "h-6 w-6",
    };

    const variantClasses = {
        default: "bg-zinc-100 text-zinc-800 dark:bg-zinc-700 dark:text-zinc-200",
        primary: "bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200",
        success: "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200",
        warning: "bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200",
        error: "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200",
    };

    const deleteButtonVariants = {
        default: "hover:bg-zinc-200 dark:hover:bg-zinc-600",
        primary: "hover:bg-blue-200 dark:hover:bg-blue-800",
        success: "hover:bg-green-200 dark:hover:bg-green-800",
        warning: "hover:bg-yellow-200 dark:hover:bg-yellow-800",
        error: "hover:bg-red-200 dark:hover:bg-red-800",
    };

    return (
        <span
            className={`
                inline-flex items-center rounded-full font-medium
                ${sizeClasses[size]}
                ${variantClasses[variant]}
                ${disabled ? "opacity-50 cursor-not-allowed" : ""}
                ${className}
            `}
        >
            {/* Icon or Image on the left */}
            {icon && (
                <span className={`flex-shrink-0 ${iconSizeClasses[size]}`}>
                    {icon}
                </span>
            )}
            {image && !icon && (
                <img
                    src={image}
                    alt=""
                    className={`flex-shrink-0 rounded-full object-cover ${imageSizeClasses[size]}`}
                />
            )}

            {/* Label */}
            <span className="truncate">{label}</span>

            {/* Delete button */}
            {onDelete && !disabled && (
                <button
                    type="button"
                    onClick={onDelete}
                    className={`
                        flex-shrink-0 rounded-full p-0.5 transition-colors cursor-pointer
                        ${deleteButtonVariants[variant]}
                        focus:outline-none focus:ring-2 focus:ring-offset-1 focus:ring-current
                    `}
                    aria-label={`Remove ${label}`}
                >
                    <XMarkIcon className={iconSizeClasses[size]} />
                </button>
            )}
        </span>
    );
}

/**
 * Container for multiple chips
 */
export function ChipGroup({ children, className = "" }) {
    return (
        <div className={`flex flex-wrap gap-2 ${className}`}>
            {children}
        </div>
    );
}
