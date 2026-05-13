import { describe, it, expect } from 'vitest';
import { formatVideoLength } from '@/utils/format';

describe('formatVideoLength', () => {
    // ── Basic seconds-only display (mm:ss) ──────────────────────────────────

    it('TC-FVL-01: formats exactly 0 seconds as "0:00"', () => {
        expect(formatVideoLength(0)).toBe('0:00');
    });

    it('TC-FVL-02: formats 59 seconds as "0:59"', () => {
        expect(formatVideoLength(59)).toBe('0:59');
    });

    it('TC-FVL-03: formats exactly 60 seconds as "1:00"', () => {
        expect(formatVideoLength(60)).toBe('1:00');
    });

    it('TC-FVL-04: formats 90 seconds (1 min 30 sec) as "1:30"', () => {
        expect(formatVideoLength(90)).toBe('1:30');
    });

    it('TC-FVL-05: formats 599 seconds as "9:59"', () => {
        expect(formatVideoLength(599)).toBe('9:59');
    });

    it('TC-FVL-06: single-digit seconds are zero-padded ("5:05")', () => {
        expect(formatVideoLength(305)).toBe('5:05');
    });

    // ── Hours display (hh:mm:ss) ─────────────────────────────────────────────

    it('TC-FVL-07: formats exactly 3600 seconds as "1:00:00"', () => {
        expect(formatVideoLength(3600)).toBe('1:00:00');
    });

    it('TC-FVL-08: formats 3661 seconds (1h 1m 1s) as "1:01:01"', () => {
        expect(formatVideoLength(3661)).toBe('1:01:01');
    });

    it('TC-FVL-09: minutes are zero-padded in h:mm:ss format', () => {
        expect(formatVideoLength(3605)).toBe('1:00:05');
    });

    it('TC-FVL-10: fractional seconds are floored, not rounded', () => {
        // 61.9 seconds → 1 min 1 sec, not 1 min 2 sec
        expect(formatVideoLength(61.9)).toBe('1:01');
    });
});
