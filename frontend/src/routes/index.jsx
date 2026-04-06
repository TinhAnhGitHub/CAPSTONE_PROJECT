import { createBrowserRouter } from "react-router-dom";
import App from "../App";
import publicRoutes from "./publicRoutes";
import privateRoutes from "./privateRoutes";

const router = createBrowserRouter([
    {
        path: "/",
        element: (
            <App />
        ),
        children: [
            ...publicRoutes,
            ...privateRoutes,
        ],
    },
]);

export default router;
