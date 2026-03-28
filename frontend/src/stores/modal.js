import { create } from 'zustand';

/**
 * Modal Store - Centralized modal state management
 * 
 * Usage:
 *   const { openModal, closeModal, getModal } = useModalStore();
 *   
 *   // Open a modal with optional data
 *   openModal('video', { image: imageData });
 *   
 *   // Close a modal
 *   closeModal('video');
 *   
 *   // Get modal state
 *   const { isOpen, data } = getModal('video');
 */
export const useModalStore = create((set, get) => ({
    modals: {},

    openModal: (name, data = null) =>
        set((state) => ({
            modals: {
                ...state.modals,
                [name]: { isOpen: true, data },
            },
        })),

    closeModal: (name) =>
        set((state) => ({
            modals: {
                ...state.modals,
                [name]: { isOpen: false, data: null },
            },
        })),

    toggleModal: (name, data = null) =>
        set((state) => {
            const current = state.modals[name];
            const isCurrentlyOpen = current?.isOpen ?? false;
            return {
                modals: {
                    ...state.modals,
                    [name]: { isOpen: !isCurrentlyOpen, data: isCurrentlyOpen ? null : data },
                },
            };
        }),

    getModal: (name) => {
        const modal = get().modals[name];
        return modal ?? { isOpen: false, data: null };
    },

    closeAllModals: () =>
        set((state) => {
            const closedModals = {};
            Object.keys(state.modals).forEach((key) => {
                closedModals[key] = { isOpen: false, data: null };
            });
            return { modals: closedModals };
        }),
}));

// Modal names constants for type safety
export const MODAL_NAMES = {
    VIDEO: 'video',
    LIBRARY: 'library',
    CONFIRM: 'confirm',
    SETTINGS: 'settings',
};
