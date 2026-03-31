import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  /** Merges conditional class names and resolves Tailwind conflicts into one string. */
  return twMerge(clsx(inputs));
}
