import { describe, it, expect } from 'vitest';
import mergeBlock from '@/utils/chat/mergeBlock';

describe('mergeBlock', () => {
    // ── null guards ───────────────────────────────────────────────────────────

    it('TC-MB-01: returns null when lastBlock is null', () => {
        expect(mergeBlock(null, { block_type: 'text', text: 'hi' })).toBeNull();
    });

    it('TC-MB-02: returns null when newBlock is null', () => {
        expect(mergeBlock({ block_type: 'text', text: 'hi' }, null)).toBeNull();
    });

    // ── text merge ────────────────────────────────────────────────────────────

    it('TC-MB-03: concatenates text blocks', () => {
        const result = mergeBlock(
            { block_type: 'text', text: 'Hello ' },
            { block_type: 'text', text: 'world' },
        );
        expect(result).toEqual({ block_type: 'text', text: 'Hello world' });
    });

    it('TC-MB-04: text merge returns a new object (immutability)', () => {
        const last = { block_type: 'text', text: 'A' };
        const merged = mergeBlock(last, { block_type: 'text', text: 'B' });
        expect(merged).not.toBe(last); // different reference
        expect(last.text).toBe('A');   // original unchanged
    });

    // ── image merge ───────────────────────────────────────────────────────────

    it('TC-MB-05: concatenates image url arrays', () => {
        const result = mergeBlock(
            { block_type: 'image', url: ['img1.jpg'] },
            { block_type: 'image', url: ['img2.jpg', 'img3.jpg'] },
        );
        expect(result.url).toEqual(['img1.jpg', 'img2.jpg', 'img3.jpg']);
    });

    // ── video merge ───────────────────────────────────────────────────────────

    it('TC-MB-06: merges video segments when same video_id', () => {
        const result = mergeBlock(
            { block_type: 'video', video_id: 'v1', segments: [{ start: 0, end: 5 }] },
            { block_type: 'video', video_id: 'v1', segments: [{ start: 10, end: 15 }] },
        );
        expect(result.segments).toHaveLength(2);
    });

    it('TC-MB-07: does NOT merge videos with different video_ids (returns null)', () => {
        const result = mergeBlock(
            { block_type: 'video', video_id: 'v1', segments: [] },
            { block_type: 'video', video_id: 'v2', segments: [] },
        );
        expect(result).toBeNull();
    });

    // ── thinking merge ────────────────────────────────────────────────────────

    it('TC-MB-08: merges thinking steps arrays', () => {
        const result = mergeBlock(
            { block_type: 'thinking', steps: [{ title: 'A' }] },
            { block_type: 'thinking', steps: [{ title: 'B' }] },
        );
        expect(result.steps).toHaveLength(2);
    });

    // ── cross-type ────────────────────────────────────────────────────────────

    it('TC-MB-09: returns null for mismatched block types', () => {
        expect(mergeBlock(
            { block_type: 'text', text: 'hi' },
            { block_type: 'image', url: [] },
        )).toBeNull();
    });
});
