import mergeBlock from "./mergeBlock";

export default function addBlocksToMessages(messages, role, newBlocks) {
    let updatedMessages = messages;
    for (const newBlock of newBlocks) {
        updatedMessages = addBlockToMessages(updatedMessages, role, newBlock);
    }
    return updatedMessages;
}

export function addBlockToMessages(messages, role, newBlock) {
    const lastMessage = messages[messages.length - 1];

    if (lastMessage && lastMessage.role === role) {
        const newMessages = [...messages];
        const messagesCopy = { ...lastMessage };
        const blocksCopy = [...messagesCopy.blocks];

        const lastBlock = blocksCopy[blocksCopy.length - 1];
        const mergedBlock = mergeBlock(lastBlock, newBlock);

        if (mergedBlock) {
            // Replace last block with merged block (new object for React)
            blocksCopy[blocksCopy.length - 1] = mergedBlock;
        } else {
            blocksCopy.push(newBlock);
        }

        messagesCopy.blocks = blocksCopy;
        newMessages[newMessages.length - 1] = messagesCopy;
        return newMessages;
    }
    return [...messages, { role, blocks: [newBlock] }];
}

export function updateToolCallBlock(messages, finished_tool_name) {
    for (let i = 0; i < messages.length; i++) {
        const message = messages[i];
        if (message.role !== 'assistant') continue;
        const blocks = message.blocks;
        for (let j = blocks.length - 1; j >= 0; j--) {
            const block = blocks[j];
            if (block.block_type === 'tool_call') {
                const steps = block.steps;
                for (let k = steps.length - 1; k >= 0; k--) {
                    const step = steps[k];
                    if (step.tool_name === finished_tool_name && step.status !== 'finished') {
                        // Update status to 'finished'
                        const updatedStep = { ...step, status: 'finished' };
                        const updatedSteps = [...steps];
                        updatedSteps[k] = updatedStep;
                        const updatedBlock = { ...block, steps: updatedSteps };
                        const updatedBlocks = [...blocks];
                        updatedBlocks[j] = updatedBlock;
                        const updatedMessage = { ...message, blocks: updatedBlocks };
                        const updatedMessages = [...messages];
                        updatedMessages[i] = updatedMessage;
                        return updatedMessages;
                    }
                }
            }
        }
    }
    return messages;
}