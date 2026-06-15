from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class AgentStep(BaseModel):
    id: str
    text: str


class SourceItem(BaseModel):
    id: str
    title: str
    excerpt: str
    meta: str
    ticker: str
    section: str


class ReportArtifact(BaseModel):
    id: str
    name: str
    size: str
    type: str
    download_url: str | None = Field(default=None, alias="downloadUrl")

    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)


class StatItem(BaseModel):
    id: str
    label: str
    value: str


class TradeProposal(BaseModel):
    ticker: str
    side: str
    quantity: int | float | str
    order_type: str = Field(alias="orderType")
    limit_price: str | None = Field(default=None, alias="limitPrice")
    risk_level: str = Field(alias="riskLevel")
    justification: str
    compliance_verdict: str | None = Field(default=None, alias="complianceVerdict")
    compliance_detail: str | None = Field(default=None, alias="complianceDetail")

    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)


class MessageArtifacts(BaseModel):
    steps: list[AgentStep] = []
    sources: list[SourceItem] = []
    reports: list[ReportArtifact] = []
    stats: list[StatItem] = []
    trade: TradeProposal | None = None

    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)
