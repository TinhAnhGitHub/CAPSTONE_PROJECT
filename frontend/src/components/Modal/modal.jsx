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
    zIndex = 'z-50',
}) {
    const handleClose = closeOnOutsideClick ? onClose : () => { };

    return (
        <Dialog
            open={isOpen}
            transition
            className={`relative ${zIndex} focus:outline-none`}
            onClose={handleClose}
        >
            <div className={`fixed inset-0 ${zIndex} w-screen overflow-y-auto bg-black/50`}>
                <div className="flex min-h-full items-center justify-center p-4">
                    <DialogPanel
                        transition
                        className={`
                            w-full  rounded-xl bg-surface p-6 backdrop-blur-2xl duration-200 ease-out data-closed:transform-[scale(95%)] data-closed:opacity-0
                            ${sizeClasses[size] || sizeClasses.lg}
                            ${(size === 'full' || size === 'xl') ? 'flex flex-col' : ''}
                            ${className}
                        `}
                    >
                        {showCloseButton && (
                            <button
                                onClick={onClose}
                                className="absolute top-4 right-4 p-1 rounded-lg text-text-muted hover:text-text hover:bg-surface-light transition-colors z-10 cursor-pointer"
                                aria-label="Close modal"
                            >
                                <XMarkIcon className="size-5" />
                            </button>
                        )}
                        {title ? (
                            // video name here
                            <DialogTitle className="text-xl font-semibold text-text mb-4 pr-10">
                                {title}
                            </DialogTitle>
                        ) : (
                            <div className="h-6" /> // Spacer when no title to prevent X clipping
                        )}
                        <div className={(size === 'full' || size === 'xl') ? 'flex-1 min-h-0 ' : ''}>
                            {children}
                        </div>
                    </DialogPanel>
                </div>
            </div>
        </Dialog>
    );
}

// Named export for convenience with modal store
export { Modal };
