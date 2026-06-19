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


class PricePoint(BaseModel):
    date: str
    close: float


class PriceSeriesStats(BaseModel):
    perf_pct: float | None = Field(default=None, alias="perfPct")
    vol_ann_pct: float | None = Field(default=None, alias="volAnnPct")
    max_drawdown_pct: float | None = Field(default=None, alias="maxDrawdownPct")
    close_min: float | None = Field(default=None, alias="closeMin")
    close_max: float | None = Field(default=None, alias="closeMax")
    close_last: float | None = Field(default=None, alias="closeLast")
    high_date: str | None = Field(default=None, alias="highDate")
    low_date: str | None = Field(default=None, alias="lowDate")

    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)


class PriceSeriesArtifact(BaseModel):
    id: str
    ticker: str
    start_date: str = Field(alias="startDate")
    end_date: str = Field(alias="endDate")
    points: list[PricePoint] = []
    stats: PriceSeriesStats | None = None

    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)


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
    price_charts: list[PriceSeriesArtifact] = Field(default_factory=list, alias="priceCharts")
    trade: TradeProposal | None = None

    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)
