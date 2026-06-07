import { describe, it, expect } from 'vitest';
import { errorIngested, ingested } from '@/utils/library';

describe('library utils', () => {
    // ── errorIngested ───────────────────────────────────────────────────────

    it('TC-LIB-01: errorIngested returns true for -1', () => {
        expect(errorIngested(-1)).toBe(true);
    });

    it('TC-LIB-02: errorIngested returns false for 0 (not yet started)', () => {
        expect(errorIngested(0)).toBe(false);
    });

    it('TC-LIB-03: errorIngested returns false for 50 (in progress)', () => {
        expect(errorIngested(50)).toBe(false);
    });

    it('TC-LIB-04: errorIngested returns false for 100 (complete)', () => {
        expect(errorIngested(100)).toBe(false);
    });

    // ── ingested ────────────────────────────────────────────────────────────

    it('TC-LIB-05: ingested returns true for 100', () => {
        expect(ingested(100)).toBe(true);
    });

    it('TC-LIB-06: ingested returns false for 99', () => {
        expect(ingested(99)).toBe(false);
    });

    it('TC-LIB-07: ingested returns false for 0', () => {
        expect(ingested(0)).toBe(false);
    });

    it('TC-LIB-08: ingested returns false for -1 (error state)', () => {
        expect(ingested(-1)).toBe(false);
    });
});
