// components/Modal.jsx

import { Dialog, DialogPanel, DialogTitle } from "@headlessui/react";
import { XMarkIcon } from '@heroicons/react/24/outline';

/**
 * Size variants for the modal
 * sm: Small modal for confirmations, alerts
 * md: Medium modal for forms, simple content
 * lg: Large modal for complex content (default)
 * xl: Extra large modal
 * full: Full screen modal
 */
const sizeClasses = {
    sm: 'w-full max-w-sm',
    md: 'w-full max-w-lg',
    lg: 'w-[70vw] max-w-5xl',
    xl: 'w-[85vw] max-w-7xl h-[90vh]',
    full: 'w-[95vw] h-[95vh] max-w-none',
};

export default function Modal({
    isOpen,
    onClose,
    title,
    children,
    size = 'lg',
    showCloseButton = true,
    closeOnOutsideClick = true,
    className = '',
}) {
    const handleClose = closeOnOutsideClick ? onClose : () => { };

    return (
        <Dialog
            open={isOpen}
            transition
            className="fixed inset-0 z-50 bg-black/50 flex items-center justify-center p-4 transition duration-300 ease-out data-closed:opacity-0"
            onClose={handleClose}
        >
            <DialogPanel
                className={`
                    bg-white dark:bg-gray-900 rounded-2xl shadow-xl p-6 relative
                    ${sizeClasses[size] || sizeClasses.lg}
                    ${(size === 'full' || size === 'xl') ? 'flex flex-col' : ''}
                    ${className}
                `}
            >
                {showCloseButton && (
                    <button
                        onClick={onClose}
                        className="absolute top-4 right-4 text-gray-400 hover:text-gray-600 dark:hover:text-white z-10"
                        aria-label="Close modal"
                    >
                        <XMarkIcon className="size-6" />
                    </button>
                )}
                {title && (
                    <DialogTitle className="text-xl font-semibold text-gray-900 dark:text-white mb-4 pr-8">
                        {title}
                    </DialogTitle>
                )}
                <div className={(size === 'full' || size === 'xl') ? 'flex-1 min-h-0 overflow-hidden' : ''}>
                    {children}
                </div>
            </DialogPanel>
        </Dialog>
    );
}

// Named export for convenience with modal store
export { Modal };
