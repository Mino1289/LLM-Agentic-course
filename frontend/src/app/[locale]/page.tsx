import { setRequestLocale } from "next-intl/server";
import { AppShell } from "@/components/layout/AppShell";

type Props = {
  params: Promise<{ locale: string }>;
};

export default async function HomePage({ params }: Props) {
  const { locale } = await params;
  setRequestLocale(locale);

  return <AppShell key={locale} />;
}
