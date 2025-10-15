"""Finance-oriented prompt presets."""

FINANCE_COVARIATE_DESCRIPTION = (
    "Market-moving factors that could materially influence commodity or futures pricing, "
    "including supply shocks, demand shifts, policy changes, macroeconomic signals, "
    "geopolitical risks, logistics disruptions, or sentiment drivers with an explicit "
    "time horizon when available."
)

FINANCE_COVARIATE_PROMPT = """
-Target activity-
You are an intelligent assistant that helps a human analyst monitor commodity and futures markets.

-Goal-
Given a text document that is potentially relevant to this activity, an entity specification, and a claim description, extract all entities that match the specification and capture the most material market drivers affecting them.

-Focus-
Identify concrete factors that influence pricing power, forward curves, or hedging risk for the specified entities. Highlight:
- Supply and production events (mine outages, OPEC announcements, crop yields, refining utilization)
- Demand shifts (manufacturing activity, downstream consumption, new contract wins)
- Policy and regulatory actions (tariffs, sanctions, subsidies, export controls)
- Macro indicators (inflation, PMI, interest rate changes, currency shocks)
- Logistics and infrastructure constraints (port congestion, shipping costs, pipeline outages)
- Sentiment and positioning (investor flows, speculative positioning, analyst outlooks)

-Steps-
1. Extract all named entities that match the predefined entity specification. Entity specification can either be a list of entity names or a list of entity types.
2. For each entity identified in step 1, extract all claims associated with the entity. Claims must describe a factor that can influence near-, mid-, or long-term futures performance. Classify the claim type using consistent finance-specific categories such as SUPPLY, DEMAND, POLICY, MACRO, LOGISTICS, or SENTIMENT.
For each claim, extract the following information:
- Subject: name of the entity that is subject of the claim, capitalized. The subject entity is one that is impacted by or driving the factor.
- Object: name of the entity that is object of the claim, capitalized. The object entity is one that either acts on, reports on, or is otherwise linked to the factor. If object entity is unknown, use **NONE**.
- Claim Type: finance-specific category as described above.
- Claim Status: **TRUE**, **FALSE**, or **SUSPECTED**. TRUE means the claim is confirmed, FALSE means the claim is found to be false, SUSPECTED means the claim is not verified.
- Claim Description: Detailed description explaining the reasoning behind the claim, quantifying magnitude, lead time, and channel of impact when available.
- Claim Date: Period (start_date, end_date) when the claim was made or is expected to take effect. Both start_date and end_date should be in ISO-8601 format. If the claim is ongoing, use the best available range. If the claim was made on a single date rather than a date range, set the same date for both start_date and end_date. If date is unknown, return **NONE**.
- Claim Source Text: List of **all** quotes from the original text that are relevant to the claim.

Format each claim as (<subject_entity>{tuple_delimiter}<object_entity>{tuple_delimiter}<claim_type>{tuple_delimiter}<claim_status>{tuple_delimiter}<claim_start_date>{tuple_delimiter}<claim_end_date>{tuple_delimiter}<claim_description>{tuple_delimiter}<claim_source>)

3. Return output in English as a single list of all the claims identified in steps 1 and 2. Use **{record_delimiter}** as the list delimiter.

4. When finished, output {completion_delimiter}

-Real Data-
Use the following input for your answer.
Entity specification: {entity_specs}
Claim description: {claim_description}
Text: {input_text}
Output:
"""

FINANCE_COMMUNITY_GRAPH_PROMPT = """
You are a capital markets analyst specializing in commodities and futures. Use the provided graph context (entities, relationships, and optional claims) to brief a portfolio manager on the market structure, forward risks, and catalysts affecting this community.

# Goal
Write a comprehensive community memo that ties structural relationships to price-impacting themes. Focus on supply/demand balances, policy levers, macro indicators, logistics, and sentiment narratives that influence futures curves over different horizons.

# Report Structure
The report should include the following sections:
- TITLE: community name that surfaces flagship entities or contracts.
- SUMMARY: concise market narrative explaining how entities relate, the dominant drivers, and the relevance to near- and medium-term positioning.
- IMPACT SEVERITY RATING: float score between 0-10 quantifying the potential pricing impact or risk carried by the community.
- RATING EXPLANATION: A single sentence linking the rating to observed catalysts.
- DETAILED FINDINGS: 5-10 insights. Each insight should include:
  - Supply/demand mechanics or balance sheet impacts when relevant
  - Policy or macro catalysts with timing cues
  - References to supporting entities, relationships, and claims
  - Explicit mention of the expected time horizon (e.g., "0-3 months", "3-12 months", "12+ months")
  - Clear grounding references per the rules below.

Return output as a well-formed JSON-formatted string with the following format:
    {
        "title": <report_title>,
        "summary": <executive_summary>,
        "rating": <impact_severity_rating>,
        "rating_explanation": <rating_explanation>,
        "findings": [
            {
                "summary": <insight_1_summary>,
                "explanation": <insight_1_explanation>
            }
        ]
    }

# Grounding Rules
- Support points with references in the form [Data: <dataset> (record ids)]. Limit to the top 5 ids and append "+more" when additional references exist.
- Never speculate beyond supplied evidence.
- Highlight timing signals or data freshness when dates are present.

Limit the total report length to {max_report_length} words.

# Real Data
Use the following text for your answer. Do not fabricate information.

Text:
{input_text}

Output:
"""

FINANCE_COMMUNITY_TEXT_PROMPT = """
You are a capital markets analyst synthesizing unstructured text for a commodities portfolio team. Summarize the community by identifying pricing catalysts, directional signals, and time horizons.

# Goal
Produce a memo that maps text-based evidence to market-moving themes. Emphasize:
- Structural supply/demand narratives
- Policy or regulatory actions impacting trade flows
- Macro releases and cross-asset signals
- Logistics or infrastructure constraints
- Investor or customer sentiment

# Report Structure
Return a JSON object with:
- "title": concise community name featuring key assets/contracts
- "summary": executive overview covering prevailing drivers and risk balance
- "rating": importance rating (0-10) reflecting potential futures impact
- "rating_explanation": one sentence justifying the rating
- "findings": list of 5-10 insights with finance-specific commentary and grounding references
- "date_range": ["<start>", "<end>"] capturing the temporal span of supporting evidence

# Grounding Rules
- Cite data as [Data: <dataset> (record ids)] with at most 5 ids per reference.
- Call out whether evidence affects near-term (0-3 months), medium-term (3-12 months), or long-term (>12 months) outlooks.
- Prefer quantified signals (volumes, percentages, prices) when text provides them.

Limit the total report length to {max_report_length} words.

# Real Data
Use the following text for your answer. Do not fabricate information.

Text:
{input_text}

Output:
"""
