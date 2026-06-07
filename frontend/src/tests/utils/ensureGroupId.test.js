import { describe, it, expect } from 'vitest';
import { ensureGroupId } from '@/utils/ensure/ensureGroupId';

describe('ensureGroupId', () => {
    it('TC-EG-01: returns ok status when currentGroup is already set', () => {
        const result = ensureGroupId([], 'group-123', () => {});
        expect(result).toEqual({ status: 'ok', group_id: 'group-123' });
    });

    it('TC-EG-02: switches to first group when currentGroup is null and groups exist', () => {
        const setCurrent = (id) => { /* side-effect stub */ };
        const groups = [{ _id: 'g1' }, { _id: 'g2' }];
        const result = ensureGroupId(groups, null, setCurrent);
        expect(result).toEqual({ status: 'switched', group_id: 'g1' });
    });

    it('TC-EG-03: calls setCurrentGroup with first group id when switching', () => {
        const calls = [];
        const groups = [{ _id: 'first-group' }];
        ensureGroupId(groups, null, (id) => calls.push(id));
        expect(calls).toEqual(['first-group']);
    });

    it('TC-EG-04: returns create status when no group set and no groups available', () => {
        const result = ensureGroupId([], null, () => {});
        expect(result).toEqual({ status: 'create' });
    });

    it('TC-EG-05: does NOT call setCurrentGroup when currentGroup is already set', () => {
        const calls = [];
        ensureGroupId([{ _id: 'g1' }], 'existing-id', (id) => calls.push(id));
        expect(calls).toHaveLength(0);
    });
});
