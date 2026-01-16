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