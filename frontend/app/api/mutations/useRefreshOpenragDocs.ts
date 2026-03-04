"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";

interface RefreshOpenRAGDocsResponse {
  message: string;
  refreshed: boolean;
}

const refreshOpenragDocs = async (): Promise<RefreshOpenRAGDocsResponse> => {
  const response = await fetch("/api/openrag-docs/refresh", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || error.error || "Failed to refresh OpenRAG docs");
  }

  return response.json();
};

export const useRefreshOpenragDocs = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: refreshOpenragDocs,
    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: ["tasks"], exact: false });
      queryClient.invalidateQueries({ queryKey: ["search"], exact: false });
      queryClient.invalidateQueries({ queryKey: ["settings"], exact: false });
    },
  });
};
