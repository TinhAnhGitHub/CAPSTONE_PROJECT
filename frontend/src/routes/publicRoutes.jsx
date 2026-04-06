import { lazy } from "react";
import Home from "@/pages/Home";
import Login from "@/pages/Login";
import ErrorBoundary from "@/components/common/errors/ErrorBoundary";

function BrokenComponent() {
    throw new Response("Broken!", { status: 500 });
}

const publicRoutes = [
    {
        path: "/",
        element: <Home />,
        errorElement: <ErrorBoundary />,
    },
    {
        path: "login",
        element: <Login />,
        errorElement: <ErrorBoundary />,
    },
    {
        path: "500",
        element: <BrokenComponent />,
        errorElement: <ErrorBoundary />,
    },
    {
        path: "*",
        element: <div>Not Found</div>,
    }
];

export default publicRoutes;
