// type ChatMessage = {
//     id?: string;       // optional unique ID for tracking
//     role: 'user' | 'assistant' | 'system';
//     content: string;
//     ts?: number;       // optional timestamp
// };

export const messages = [
    {
        id: "1",
        role: 'user',
        content: "Hello, how are you?",
        ts: 1696118400000 // Example timestamp
    },
    {
        id: "2",
        role: 'assistant',
        content: "I'm fine, thank you! How can I assist you today?",
        ts: 1696118460000 // Example timestamp
    },
    {
        id: "3",
        role: 'user',
        content: "What is the weather like today?",
        ts: 1696118520000 // Example timestamp
    },
    {
        id: "4",
        role: 'assistant',
        content: "The weather is sunny with a high of 75°F (24°C).",
        ts: 1696118580000 // Example timestamp
    }
];

export const messagesConversations = [
    {
        id: "conv1",
        title: "Chat about Weather",
    },
    {
        id: "conv2",
        title: "General Questions",
    },
    {
        id: "conv3",
        title: "Tech Support",
    },
    {
        id: "conv3",
        title: "Random Chat",
    },
]
 
import { RANDOM_IMAGE_URLS } from "@/constants/image";

export const videosConversations = [
    {
        videoId: "vid1",
        title: "react.mp4",
        length: "10:05",
        imageUrl: RANDOM_IMAGE_URLS()
    },
    {
        videoId: "vid2",
        title: "user_guide.mp4",
        length: "15:30",
        imageUrl: RANDOM_IMAGE_URLS()
    },
    {
        videoId: "vid3",
        title: "css_flexbox.mp4",
        length: "8:45",
        imageUrl: RANDOM_IMAGE_URLS()   
    },
]

export const userVideos = [
    {
        videoId: "vid1",
        title: "react.mp4",
        length: "10:05",
        imageUrl: RANDOM_IMAGE_URLS()
    },
    {
        videoId: "vid2",
        title: "user_guide.mp4",
        length: "15:30",
        imageUrl: RANDOM_IMAGE_URLS()
    },
    {
        videoId: "vid3",
        title: "css_flexbox.mp4",
        length: "8:45",
        imageUrl: RANDOM_IMAGE_URLS()
    },
    {
        videoId: "vid4",
        title: "cat.mp4",
        length: "12:34",
        imageUrl: RANDOM_IMAGE_URLS()
    },
    {
        videoId: "vid5",
        title: "dog.mp4",
        length: "10:00",
        imageUrl: RANDOM_IMAGE_URLS()
    },
]

export const VideosInConversation = [
    {
        videoId: "vid1",
        title: "react.mp4",
        length: "10:05",
        imageUrl: RANDOM_IMAGE_URLS(),
        selected: true
    },
    {
        videoId: "vid2",
        title: "user_guide.mp4",
        length: "15:30",
        imageUrl: RANDOM_IMAGE_URLS(),
        selected: true
    },
    {
        videoId: "vid3",
        title: "css_flexbox.mp4",
        length: "8:45",
        imageUrl: RANDOM_IMAGE_URLS(),
        selected: false
    },
    {
        videoId: "vid4",
        title: "cat.mp4",
        length: "12:34",
        imageUrl: RANDOM_IMAGE_URLS(),
        selected: true
    },
    {
        videoId: "vid5",
        title: "dog.mp4",
        length: "10:00",
        imageUrl: RANDOM_IMAGE_URLS(),
        selected: false
    },
]