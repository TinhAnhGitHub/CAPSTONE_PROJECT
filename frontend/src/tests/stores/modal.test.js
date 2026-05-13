import { describe, it, expect, beforeEach } from 'vitest';
import { useModalStore, MODAL_NAMES } from '@/stores/modal';

// Reset store state between tests
beforeEach(() => {
    useModalStore.setState({ modals: {} });
});

describe('useModalStore', () => {
    it('TC-MS-01: initial modals state is an empty object', () => {
        expect(useModalStore.getState().modals).toEqual({});
    });

    it('TC-MS-02: openModal sets isOpen=true with no data', () => {
        useModalStore.getState().openModal('video');
        const { isOpen, data } = useModalStore.getState().getModal('video');
        expect(isOpen).toBe(true);
        expect(data).toBeNull();
    });

    it('TC-MS-03: openModal stores provided data', () => {
        const payload = { url: 'http://example.com/video.mp4' };
        useModalStore.getState().openModal('video', payload);
        const { data } = useModalStore.getState().getModal('video');
        expect(data).toEqual(payload);
    });

    it('TC-MS-04: closeModal sets isOpen=false and clears data', () => {
        useModalStore.getState().openModal('video', { foo: 'bar' });
        useModalStore.getState().closeModal('video');
        const { isOpen, data } = useModalStore.getState().getModal('video');
        expect(isOpen).toBe(false);
        expect(data).toBeNull();
    });

    it('TC-MS-05: toggleModal opens a closed modal', () => {
        useModalStore.getState().toggleModal('settings');
        expect(useModalStore.getState().getModal('settings').isOpen).toBe(true);
    });

    it('TC-MS-06: toggleModal closes an open modal and clears data', () => {
        useModalStore.getState().openModal('settings', { x: 1 });
        useModalStore.getState().toggleModal('settings');
        const modal = useModalStore.getState().getModal('settings');
        expect(modal.isOpen).toBe(false);
        expect(modal.data).toBeNull();
    });

    it('TC-MS-07: getModal returns {isOpen:false, data:null} for unknown modal', () => {
        const modal = useModalStore.getState().getModal('nonexistent');
        expect(modal).toEqual({ isOpen: false, data: null });
    });

    it('TC-MS-08: closeAllModals closes every open modal', () => {
        useModalStore.getState().openModal('video');
        useModalStore.getState().openModal('settings');
        useModalStore.getState().closeAllModals();
        const v = useModalStore.getState().getModal('video');
        const s = useModalStore.getState().getModal('settings');
        expect(v.isOpen).toBe(false);
        expect(s.isOpen).toBe(false);
    });

    it('TC-MS-09: multiple modals can be open independently', () => {
        useModalStore.getState().openModal(MODAL_NAMES.VIDEO);
        useModalStore.getState().openModal(MODAL_NAMES.LIBRARY);
        expect(useModalStore.getState().getModal(MODAL_NAMES.VIDEO).isOpen).toBe(true);
        expect(useModalStore.getState().getModal(MODAL_NAMES.LIBRARY).isOpen).toBe(true);
    });

    it('TC-MS-10: MODAL_NAMES constants match expected keys', () => {
        expect(MODAL_NAMES.VIDEO).toBe('video');
        expect(MODAL_NAMES.LIBRARY).toBe('library');
        expect(MODAL_NAMES.CONFIRM).toBe('confirm');
        expect(MODAL_NAMES.SETTINGS).toBe('settings');
    });
});
