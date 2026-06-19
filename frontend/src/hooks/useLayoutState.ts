"use client";

import { useState } from "react";

export function useDisclosure(initial = false) {
  const [isOpen, setIsOpen] = useState(initial);
  return {
    isOpen,
    open: () => setIsOpen(true),
    close: () => setIsOpen(false),
    toggle: () => setIsOpen((prev) => !prev),
  };
}

export function useSidebarState() {
  const [collapsed, setCollapsed] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);

  const toggleSidebar = () => {
    if (typeof window !== "undefined" && window.innerWidth <= 920) {
      setMobileOpen((prev) => !prev);
      return;
    }
    setCollapsed((prev) => !prev);
  };

  const closeMobile = () => setMobileOpen(false);

  return { collapsed, mobileOpen, toggleSidebar, closeMobile };
}
