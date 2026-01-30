from gym.envs.registration import register

from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv

# Browser 环境需要 selenium，延迟导入以避免不必要的依赖
# from web_agent_site.envs.web_agent_site_env import WebAgentSiteEnv

register(
  id='WebAgentTextEnv-v0',
  entry_point='web_agent_site.envs:WebAgentTextEnv',
)

register(
  id='WebAgentSiteEnv-v0',
  entry_point='web_agent_site.envs.web_agent_site_env:WebAgentSiteEnv',
)