import { describe, it, expect } from 'vitest';
import parseChunkToBlock from '@/utils/chat/parseChunkToBlock';

describe('parseChunkToBlock', () => {
    // ── text ─────────────────────────────────────────────────────────────────

    it('TC-PCB-01: text chunk returns a text block', () => {
        const result = parseChunkToBlock('text', 'Hello');
        expect(result).toEqual({ block_type: 'text', text: 'Hello' });
    });

    it('TC-PCB-02: empty text chunk defaults to a single space', () => {
        const result = parseChunkToBlock('text', '');
        expect(result).toEqual({ block_type: 'text', text: ' ' });
    });

    it('TC-PCB-03: null text chunk defaults to a single space', () => {
        const result = parseChunkToBlock('text', null);
        expect(result).toEqual({ block_type: 'text', text: ' ' });
    });

    // ── image ─────────────────────────────────────────────────────────────────

    it('TC-PCB-04: image chunk extracts url from each item', () => {
        const chunk = [{ url: 'http://a.com/1.jpg' }, { url: 'http://a.com/2.jpg' }];
        const [result] = parseChunkToBlock('image', chunk);
        expect(result.block_type).toBe('image');
        expect(result.url).toEqual(['http://a.com/1.jpg', 'http://a.com/2.jpg']);
    });

    it('TC-PCB-05: image result is always wrapped in an array', () => {
        const result = parseChunkToBlock('image', [{ url: 'http://x.com/img.jpg' }]);
        expect(Array.isArray(result)).toBe(true);
    });

    // ── video ─────────────────────────────────────────────────────────────────

    it('TC-PCB-06: video chunk is returned as-is', () => {
        const chunk = { video_id: 'abc', segments: [{ start: 0, end: 5 }] };
        expect(parseChunkToBlock('video', chunk)).toBe(chunk);
    });

    // ── thinking ─────────────────────────────────────────────────────────────

    it('TC-PCB-07: thinking chunk wraps step in steps array', () => {
        const chunk = { title: 'step1', content: 'reasoning' };
        const result = parseChunkToBlock('thinking', chunk);
        expect(result.block_type).toBe('thinking');
        expect(result.steps).toHaveLength(1);
        expect(result.steps[0]).toMatchObject(chunk);
    });

    // ── tools ─────────────────────────────────────────────────────────────────

    it('TC-PCB-08: tools chunk has status pending by default', () => {
        const chunk = { tool_name: 'search_video' };
        const result = parseChunkToBlock('tools', chunk);
        expect(result.block_type).toBe('tools');
        expect(result.steps[0].status).toBe('pending');
        expect(result.steps[0].tool_name).toBe('search_video');
    });

    // ── unknown ───────────────────────────────────────────────────────────────

    it('TC-PCB-09: unknown msg_type returns null', () => {
        expect(parseChunkToBlock('unknown_type', {})).toBeNull();
    });
});
