"""40 prompts across 4 SaaS categories for LightGBM training data generation."""
from src.utils.models import PromptConfig

PM_BRANDS = ["Asana", "Jira", "Linear", "Monday.com", "Notion", "ClickUp", "Trello", "Basecamp"]
CRM_BRANDS = ["HubSpot", "Salesforce", "Pipedrive", "Attio", "Zoho CRM", "Close", "Copper"]
AI_WRITING_BRANDS = ["Notion AI", "Jasper", "Copy.ai", "Grammarly", "Writer", "Writesonic", "Rytr"]
DEV_TOOLS_BRANDS = ["GitHub", "GitLab", "Linear", "Jira", "Shortcut", "Azure DevOps", "YouTrack"]

BATCH_PROMPTS: list[PromptConfig] = [

    # ── Project Management (10 prompts) ───────────────────────────────────────
    PromptConfig(
        prompt_id="pm_b01",
        prompt_text="What is the best project management tool for a 10-person startup? Please cite your sources with URLs.",
        category="project_management",
        target_brands=PM_BRANDS,
    ),
    PromptConfig(
        prompt_id="pm_b02",
        prompt_text="Compare Notion vs Linear for engineering teams. Which is better and why? Please cite sources.",
        category="project_management",
        target_brands=PM_BRANDS,
    ),
    PromptConfig(
        prompt_id="pm_b03",
        prompt_text="What are the most affordable project management tools for small teams in 2025? Please cite sources.",
        category="project_management",
        target_brands=PM_BRANDS,
    ),
    PromptConfig(
        prompt_id="pm_b04",
        prompt_text="Which project management software is best for Agile development teams? Please cite sources.",
        category="project_management",
        target_brands=PM_BRANDS,
    ),
    PromptConfig(
        prompt_id="pm_b05",
        prompt_text="What project management tool should I use instead of Jira? Please cite your sources.",
        category="project_management",
        target_brands=PM_BRANDS,
    ),
    PromptConfig(
        prompt_id="pm_b06",
        prompt_text="Best project management tools for non-technical teams in 2025. Please cite sources with URLs.",
        category="project_management",
        target_brands=PM_BRANDS,
    ),
    PromptConfig(
        prompt_id="pm_b07",
        prompt_text="What is the difference between Asana and Monday.com? Which should I choose? Please cite sources.",
        category="project_management",
        target_brands=PM_BRANDS,
    ),
    PromptConfig(
        prompt_id="pm_b08",
        prompt_text="Top project management platforms for enterprise companies. Please cite your sources.",
        category="project_management",
        target_brands=PM_BRANDS,
    ),
    PromptConfig(
        prompt_id="pm_b09",
        prompt_text="What are the best free project management tools available in 2025? Please cite sources.",
        category="project_management",
        target_brands=PM_BRANDS,
    ),
    PromptConfig(
        prompt_id="pm_b10",
        prompt_text="Which project management tool has the best Slack and GitHub integrations? Please cite sources.",
        category="project_management",
        target_brands=PM_BRANDS,
    ),

    # ── CRM (10 prompts) ──────────────────────────────────────────────────────
    PromptConfig(
        prompt_id="crm_b01",
        prompt_text="What is the best CRM for B2B SaaS startups in 2025? Please cite your sources with URLs.",
        category="crm",
        target_brands=CRM_BRANDS,
    ),
    PromptConfig(
        prompt_id="crm_b02",
        prompt_text="What is the simplest CRM for a small sales team of 5 people? Please cite sources.",
        category="crm",
        target_brands=CRM_BRANDS,
    ),
    PromptConfig(
        prompt_id="crm_b03",
        prompt_text="Compare HubSpot vs Salesforce for a growing startup. Which is better? Please cite sources.",
        category="crm",
        target_brands=CRM_BRANDS,
    ),
    PromptConfig(
        prompt_id="crm_b04",
        prompt_text="What are the best free CRM tools for small businesses? Please cite your sources.",
        category="crm",
        target_brands=CRM_BRANDS,
    ),
    PromptConfig(
        prompt_id="crm_b05",
        prompt_text="Which CRM software is best for managing outbound sales pipelines? Please cite sources.",
        category="crm",
        target_brands=CRM_BRANDS,
    ),
    PromptConfig(
        prompt_id="crm_b06",
        prompt_text="Best CRM alternatives to Salesforce for mid-size companies. Please cite sources with URLs.",
        category="crm",
        target_brands=CRM_BRANDS,
    ),
    PromptConfig(
        prompt_id="crm_b07",
        prompt_text="What CRM should a founder use when first building a sales process? Please cite sources.",
        category="crm",
        target_brands=CRM_BRANDS,
    ),
    PromptConfig(
        prompt_id="crm_b08",
        prompt_text="Which CRM has the best email and LinkedIn integration in 2025? Please cite sources.",
        category="crm",
        target_brands=CRM_BRANDS,
    ),
    PromptConfig(
        prompt_id="crm_b09",
        prompt_text="Top CRM tools for SaaS companies with product-led growth. Please cite your sources.",
        category="crm",
        target_brands=CRM_BRANDS,
    ),
    PromptConfig(
        prompt_id="crm_b10",
        prompt_text="What is the best CRM for tracking customer success and renewals? Please cite sources.",
        category="crm",
        target_brands=CRM_BRANDS,
    ),

    # ── AI Writing Tools (10 prompts) ─────────────────────────────────────────
    PromptConfig(
        prompt_id="ai_b01",
        prompt_text="What are the best AI writing tools for content marketing teams in 2025? Please cite sources.",
        category="ai_writing",
        target_brands=AI_WRITING_BRANDS,
    ),
    PromptConfig(
        prompt_id="ai_b02",
        prompt_text="Compare Jasper vs Copy.ai for writing marketing copy. Which is better? Please cite sources.",
        category="ai_writing",
        target_brands=AI_WRITING_BRANDS,
    ),
    PromptConfig(
        prompt_id="ai_b03",
        prompt_text="What is the best AI writing assistant for long-form blog content? Please cite sources.",
        category="ai_writing",
        target_brands=AI_WRITING_BRANDS,
    ),
    PromptConfig(
        prompt_id="ai_b04",
        prompt_text="Which AI writing tools are best for enterprise brand consistency? Please cite sources.",
        category="ai_writing",
        target_brands=AI_WRITING_BRANDS,
    ),
    PromptConfig(
        prompt_id="ai_b05",
        prompt_text="Best affordable AI writing tools for freelance writers in 2025. Please cite sources.",
        category="ai_writing",
        target_brands=AI_WRITING_BRANDS,
    ),
    PromptConfig(
        prompt_id="ai_b06",
        prompt_text="What AI tools help with grammar checking and writing style improvement? Please cite sources.",
        category="ai_writing",
        target_brands=AI_WRITING_BRANDS,
    ),
    PromptConfig(
        prompt_id="ai_b07",
        prompt_text="Which AI writing assistant integrates best with Notion and Google Docs? Please cite sources.",
        category="ai_writing",
        target_brands=AI_WRITING_BRANDS,
    ),
    PromptConfig(
        prompt_id="ai_b08",
        prompt_text="Top AI copywriting tools for social media and ad campaigns. Please cite sources with URLs.",
        category="ai_writing",
        target_brands=AI_WRITING_BRANDS,
    ),
    PromptConfig(
        prompt_id="ai_b09",
        prompt_text="What AI writing tools are most accurate and least likely to hallucinate? Please cite sources.",
        category="ai_writing",
        target_brands=AI_WRITING_BRANDS,
    ),
    PromptConfig(
        prompt_id="ai_b10",
        prompt_text="Best AI tools for writing technical documentation and API docs. Please cite sources.",
        category="ai_writing",
        target_brands=AI_WRITING_BRANDS,
    ),

    # ── Developer Tools (10 prompts) ──────────────────────────────────────────
    PromptConfig(
        prompt_id="dev_b01",
        prompt_text="What is the best issue tracking tool for software engineering teams in 2025? Please cite sources.",
        category="developer_tools",
        target_brands=DEV_TOOLS_BRANDS,
    ),
    PromptConfig(
        prompt_id="dev_b02",
        prompt_text="Compare GitHub vs GitLab for a self-hosted DevOps setup. Please cite sources.",
        category="developer_tools",
        target_brands=DEV_TOOLS_BRANDS,
    ),
    PromptConfig(
        prompt_id="dev_b03",
        prompt_text="What is the best Jira alternative for fast-moving engineering teams? Please cite sources.",
        category="developer_tools",
        target_brands=DEV_TOOLS_BRANDS,
    ),
    PromptConfig(
        prompt_id="dev_b04",
        prompt_text="Which developer tools are best for sprint planning and backlog management? Please cite sources.",
        category="developer_tools",
        target_brands=DEV_TOOLS_BRANDS,
    ),
    PromptConfig(
        prompt_id="dev_b05",
        prompt_text="What tools do top engineering teams use for tracking bugs and feature requests? Please cite sources.",
        category="developer_tools",
        target_brands=DEV_TOOLS_BRANDS,
    ),
    PromptConfig(
        prompt_id="dev_b06",
        prompt_text="Best project tracking tools that integrate with GitHub pull requests. Please cite sources.",
        category="developer_tools",
        target_brands=DEV_TOOLS_BRANDS,
    ),
    PromptConfig(
        prompt_id="dev_b07",
        prompt_text="Which issue tracker has the best API and automation capabilities? Please cite sources.",
        category="developer_tools",
        target_brands=DEV_TOOLS_BRANDS,
    ),
    PromptConfig(
        prompt_id="dev_b08",
        prompt_text="Top CI/CD and DevOps tools for modern software teams in 2025. Please cite sources.",
        category="developer_tools",
        target_brands=DEV_TOOLS_BRANDS,
    ),
    PromptConfig(
        prompt_id="dev_b09",
        prompt_text="What project management tool is most popular among YC-backed startups? Please cite sources.",
        category="developer_tools",
        target_brands=DEV_TOOLS_BRANDS,
    ),
    PromptConfig(
        prompt_id="dev_b10",
        prompt_text="Best tools for managing a software roadmap and communicating it to stakeholders. Please cite sources.",
        category="developer_tools",
        target_brands=DEV_TOOLS_BRANDS,
    ),
]
