import { describe, it, expect } from 'vitest';
import { ensureSessionId } from '@/utils/ensure/ensureSessionId';

describe('ensureSessionId', () => {
    const sessions = [{ _id: 's1' }, { _id: 's2' }];

    it('TC-ES-01: returns ok when session exists in sessions list', () => {
        const result = ensureSessionId(sessions, 's1', () => {});
        expect(result).toEqual({ status: 'ok', group_id: 's1' });
    });

    it('TC-ES-02: returns ok for second session id in list', () => {
        const result = ensureSessionId(sessions, 's2', () => {});
        expect(result).toEqual({ status: 'ok', group_id: 's2' });
    });

    it('TC-ES-03: switches to first session when current session is not in list', () => {
        const result = ensureSessionId(sessions, 'stale-id', () => {});
        expect(result).toEqual({ status: 'switched', group_id: 's1' });
    });

    it('TC-ES-04: switches to first session when current session is null', () => {
        const result = ensureSessionId(sessions, null, () => {});
        expect(result).toEqual({ status: 'switched', group_id: 's1' });
    });

    it('TC-ES-05: calls setSession when switching', () => {
        const calls = [];
        ensureSessionId(sessions, null, (id) => calls.push(id));
        expect(calls).toEqual(['s1']);
    });

    it('TC-ES-06: returns create status when sessions is empty', () => {
        const result = ensureSessionId([], null, () => {});
        expect(result).toEqual({ status: 'create' });
    });
});
