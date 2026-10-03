import { QueryClientProvider } from "@tanstack/react-query";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

// fonts are bundled (latin subsets): the app makes no network request outside /api
import "./styles/fonts.css";
import "@fontsource/ibm-plex-sans/latin-400.css";
import "@fontsource/ibm-plex-sans/latin-500.css";
import "@fontsource/ibm-plex-sans/latin-600.css";
import "@fontsource/ibm-plex-mono/latin-400.css";
import "@fontsource/ibm-plex-mono/latin-500.css";

import "./styles/base.css";
import "./styles/components.css";
import "./styles/views.css";
import "./styles/overview.css";
import "./styles/sections.css";

import { createQueryClient } from "./api/queries";
import { App } from "./App";
import { applyTheme } from "./lib/theme";

applyTheme();

const queryClient = createQueryClient();

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <App />
    </QueryClientProvider>
  </StrictMode>,
);
