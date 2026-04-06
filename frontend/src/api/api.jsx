import axios from "axios";
import { PRIMARY_URL } from "@/constants/url";
import { useStore } from "@/stores/user";

const api = axios.create({
    baseURL: PRIMARY_URL,
});

api.interceptors.request.use(
    (config) => {
        const token = useStore.getState().token;
        if (token) {
            config.headers['Authorization'] = `Bearer ${token}`;
        }
        else {
            delete config.headers['Authorization'];
        }
        return config;
    },
    (error) => {
        return Promise.reject(error);
    }
);

export default api;