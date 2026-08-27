from __future__ import annotations

import streamlit as st

from knowledgelens.provider_activation import active_profile_record, set_active_profile
from knowledgelens.provider_profiles import (
    KNOWN_CAPABILITIES,
    configured_deployment_mode,
    discover_models,
    profile_management_error,
)
from knowledgelens.services import build_application_services

st.set_page_config(page_title="KnowledgeLens AI · Providers", page_icon="◉", layout="wide")


@st.cache_resource
def services():
    return build_application_services()


svc = services()
st.title("AI provider profiles")
st.caption("Persist endpoints, models, capabilities, and credentials without restarting KnowledgeLens.")

if not svc.database.enabled:
    st.error("PostgreSQL is not configured. Set KNOWLEDGELENS_DATABASE_URL, restart once, then manage profiles here.")
    st.stop()
if svc.profiles is None:
    st.error(svc.persistence_error or "Persistent provider profiles are unavailable.")
    st.stop()

deployment_mode = configured_deployment_mode()
role = "admin" if deployment_mode in {"local", "private"} else None
management_error = profile_management_error(deployment_mode=deployment_mode, role=role)

try:
    profiles = svc.profiles.list()
    active = active_profile_record(svc.database)
except Exception as exc:
    st.error(f"Could not load provider profiles: {exc}")
    st.stop()

if active:
    st.success(f"Active workspace provider: {active['name']} · {active['base_url']} · {active['default_model']}")
else:
    st.info("No persistent profile is active. The workspace will use the legacy in-session provider selector.")

st.subheader("Configured profiles")
for profile in profiles:
    badges = ", ".join(profile.capabilities) or "text"
    suffix = " · built-in" if profile.is_builtin else ""
    st.write(f"**{profile.name}**{suffix}  \n`{profile.base_url}` · `{profile.default_model}` · {badges}")

profile_map = {profile.id: profile for profile in profiles}
selection = st.selectbox(
    "Edit profile",
    ["__new__", *profile_map.keys()],
    format_func=lambda value: "＋ New provider profile" if value == "__new__" else profile_map[value].name,
)
selected = profile_map.get(selection)

if management_error:
    st.warning(management_error)
    st.info(
        "PR #2 keeps public deployments read-only until the bootstrap-admin/OIDC session layer is wired in PR #5. "
        "Local/private trusted deployments can manage endpoints now."
    )
    st.stop()

if selected and st.button("Activate this profile", type="primary"):
    try:
        set_active_profile(svc.database, selected.id)
        st.session_state["model_configured"] = selected.default_model
        st.success(
            f"Activated {selected.name}. Return to the workspace and select Configured endpoint; no server restart is required."
        )
    except Exception as exc:
        st.error(f"Could not activate provider profile: {exc}")

provider_types = ["openai-compatible", "openai", "ollama", "llama.cpp"]
with st.form("provider_profile_form"):
    name = st.text_input("Profile name", value=selected.name if selected else "")
    selected_type = selected.provider_type if selected and selected.provider_type in provider_types else provider_types[0]
    provider_type = st.selectbox("Provider type", provider_types, index=provider_types.index(selected_type))
    base_url = st.text_input("Base URL", value=selected.base_url if selected else "")
    default_model = st.text_input("Default model", value=selected.default_model if selected else "")
    capabilities = st.multiselect(
        "Capabilities",
        list(KNOWN_CAPABILITIES),
        default=list(selected.capabilities if selected else ("text",)),
        help="Capabilities remain explicit unless the provider supplies trustworthy metadata; model names are never used as proof of vision support.",
    )
    credential = st.text_input(
        "API key / bearer credential",
        type="password",
        help="Leave blank to keep an existing credential. New values prefer the OS keychain and use encrypted PostgreSQL fallback on servers.",
    )
    submitted = st.form_submit_button("Save profile")

if submitted:
    try:
        secret_ref = selected.secret_ref if selected else None
        saved = svc.profiles.save(
            name=name,
            provider_type=provider_type,
            base_url=base_url,
            default_model=default_model,
            capabilities=capabilities,
            secret_ref=secret_ref,
            profile_id=selected.id if selected else None,
            is_builtin=selected.is_builtin if selected else False,
        )
        if credential:
            if svc.secrets is None:
                raise RuntimeError("No secure secret store is available.")
            secret_ref = saved.secret_ref or f"provider:{saved.id}:api-key"
            svc.secrets.set(secret_ref, credential)
            saved = svc.profiles.save(
                name=saved.name,
                provider_type=saved.provider_type,
                base_url=saved.base_url,
                default_model=saved.default_model,
                capabilities=saved.capabilities,
                secret_ref=secret_ref,
                profile_id=saved.id,
                is_builtin=saved.is_builtin,
            )
        st.success(f"Saved {saved.name}. It can be activated immediately without restarting KnowledgeLens.")
        st.rerun()
    except Exception as exc:
        st.error(f"Could not save provider profile: {exc}")

if selected:
    st.divider()
    st.subheader("Test & discover")
    st.caption("Discovery uses the same bounded DNS-pinned endpoint policy as LLM requests.")
    if st.button("Discover models"):
        api_key = ""
        if selected.secret_ref and svc.secrets is not None:
            try:
                api_key = svc.secrets.get(selected.secret_ref) or ""
            except RuntimeError as exc:
                st.error(str(exc))
        try:
            models = discover_models(selected, api_key)
        except Exception as exc:
            st.error(f"Discovery failed: {exc}")
        else:
            if models:
                st.success(f"Discovered {len(models)} model(s).")
                st.code("\n".join(models[:200]))
            else:
                st.info("The provider returned no model identifiers.")
