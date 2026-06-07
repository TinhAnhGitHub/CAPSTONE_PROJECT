import { describe, it, expect, beforeEach, vi } from 'vitest';
import { addBlockToMessages, updateToolCallBlock } from '@/utils/chat/addBlockToMessages';
import addBlocksToMessages from '@/utils/chat/addBlockToMessages';

// ── helpers ──────────────────────────────────────────────────────────────────

const textBlock = (text) => ({ block_type: 'text', text });
const imageBlock = (urls) => ({ block_type: 'image', url: urls });
const toolBlock = (tool_name, status = 'pending') => ({
    block_type: 'tools',
    steps: [{ tool_name, status }],
});
const msg = (role, blocks) => ({ role, blocks });

// ── addBlockToMessages ────────────────────────────────────────────────────────

describe('addBlockToMessages', () => {
    it('TC-ABM-01: creates new message when list is empty', () => {
        const result = addBlockToMessages([], 'assistant', textBlock('hello'));
        expect(result).toHaveLength(1);
        expect(result[0].role).toBe('assistant');
    });

    it('TC-ABM-02: merges block into last message with same role', () => {
        const initial = [msg('assistant', [textBlock('hi ')])];
        const result = addBlockToMessages(initial, 'assistant', textBlock('there'));
        expect(result).toHaveLength(1);
        expect(result[0].blocks[0].text).toBe('hi there');
    });

    it('TC-ABM-03: creates new message when role differs', () => {
        const initial = [msg('assistant', [textBlock('hi')])];
        const result = addBlockToMessages(initial, 'user', textBlock('reply'));
        expect(result).toHaveLength(2);
        expect(result[1].role).toBe('user');
    });

    it('TC-ABM-04: appends incompatible block as new block in same message', () => {
        const initial = [msg('assistant', [textBlock('hello')])];
        const result = addBlockToMessages(initial, 'assistant', imageBlock(['a.jpg']));
        expect(result[0].blocks).toHaveLength(2);
    });

    it('TC-ABM-05: does not mutate the original messages array', () => {
        const initial = [msg('assistant', [textBlock('A')])];
        const copy = JSON.stringify(initial);
        addBlockToMessages(initial, 'assistant', textBlock('B'));
        expect(JSON.stringify(initial)).toBe(copy);
    });
});

// ── addBlocksToMessages (plural) ──────────────────────────────────────────────

describe('addBlocksToMessages', () => {
    it('TC-ABMs-01: processes multiple blocks sequentially', () => {
        const result = addBlocksToMessages([], 'assistant', [
            textBlock('A'),
            textBlock('B'),
        ]);
        expect(result[0].blocks[0].text).toBe('AB');
    });

    it('TC-ABMs-02: handles empty newBlocks array without error', () => {
        const initial = [msg('assistant', [textBlock('hi')])];
        const result = addBlocksToMessages(initial, 'assistant', []);
        expect(result).toEqual(initial);
    });
});

// ── updateToolCallBlock ────────────────────────────────────────────────────────

describe('updateToolCallBlock', () => {
    it('TC-UTC-01: marks matching tool step as finished', () => {
        const messages = [msg('assistant', [toolBlock('search_video')])];
        const result = updateToolCallBlock(messages, 'search_video');
        expect(result[0].blocks[0].steps[0].status).toBe('finished');
    });

    it('TC-UTC-02: does not mutate original messages', () => {
        const messages = [msg('assistant', [toolBlock('search_video')])];
        const copy = JSON.stringify(messages);
        updateToolCallBlock(messages, 'search_video');
        expect(JSON.stringify(messages)).toBe(copy);
    });

    it('TC-UTC-03: returns original array when tool name not found', () => {
        const messages = [msg('assistant', [toolBlock('other_tool')])];
        const result = updateToolCallBlock(messages, 'search_video');
        expect(result).toBe(messages);
    });

    it('TC-UTC-04: skips user messages (only updates assistant messages)', () => {
        const messages = [msg('user', [toolBlock('search_video')])];
        const result = updateToolCallBlock(messages, 'search_video');
        // user messages are skipped — original returned unchanged
        expect(result).toBe(messages);
    });
});
