import gym
import random
import requests
import string
import time

from bs4 import BeautifulSoup
from bs4.element import Comment
from gym import spaces
from os.path import join, dirname, abspath
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import ElementNotInteractableException

# 尝试使用 webdriver-manager 自动管理 chromedriver
try:
    from webdriver_manager.chrome import ChromeDriverManager
    USE_WEBDRIVER_MANAGER = True
except ImportError:
    USE_WEBDRIVER_MANAGER = False

from web_agent_site.engine.engine import parse_action, END_BUTTON

class WebAgentSiteEnv(gym.Env):
    """Gym environment for HTML mode of WebShop environment"""

    def __init__(self, observation_mode='html', **kwargs):
        """
        Constructor for HTML environment

        Arguments:
        observation_mode (`str`) -- ['html' | 'text'] (default 'html')
        pause (`float`) -- Pause (in seconds) after taking an action. 
            This is mainly for demo purposes.
            Recommended value: 2.0s
        render (`bool`) -- Show browser if set to `True`.
        session ('str') -- Session ID to initialize environment with
        server_url (`str`) -- Server URL (default 'http://127.0.0.1:3000')
        headless (`bool`) -- Run browser in headless mode (default True if render=False)
        """
        super(WebAgentSiteEnv, self).__init__()
        self.observation_mode = observation_mode
        self.kwargs = kwargs
        self.server_url = kwargs.get('server_url', 'http://127.0.0.1:3000')

        # Create a browser driver to simulate the WebShop site
        options = Options()
        if 'render' not in kwargs or not kwargs['render']:
            options.add_argument("--headless")  # don't show browser
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
        
        # 使用 webdriver-manager 自动管理 chromedriver，或使用本地 chromedriver
        if USE_WEBDRIVER_MANAGER:
            service = Service(ChromeDriverManager().install())
        else:
            service = Service(join(dirname(abspath(__file__)), 'chromedriver'))
        
        self.browser = webdriver.Chrome(service=service, options=options)

        # Set flags and values for WebShop session
        self.text_to_clickable = None
        self.assigned_session = kwargs.get('session')
        self.session = None
        self.reset()

    def step(self, action):
        """
        Takes an action, updates WebShop environment, and returns (observation, reward, done, info)

        Arguments:
        action (`str`): An action should be of the following structure:
          - search[keywords]
          - click[value]
        If action not valid, perform nothing.
        """
        reward = 0.0
        done = False
        info = None

        # Map action to executed command on the WebShop environment via the browser driver
        action_name, action_arg = parse_action(action)
        if action_name == 'search':
            try:
                search_bar = self.browser.find_element(By.ID, 'search_input')
            except Exception:
                pass
            else:
                search_bar.send_keys(action_arg)
                search_bar.submit()
        elif action_name == 'click':
            # 支持大小写不敏感的查找
            clickable = self._find_clickable(action_arg)
            if clickable is not None:
                try:
                    clickable.click()
                except ElementNotInteractableException:
                    # Perform force click with JavaScript
                    self.browser.execute_script("arguments[0].click();", clickable)
                reward = self.get_reward()
                if action_arg.lower() == END_BUTTON.lower():
                    done = True
        elif action_name == 'end':
            done = True
        else:
            print('Invalid action. No action performed.')

        if 'pause' in self.kwargs:
            time.sleep(self.kwargs['pause'])
        return self.observation, reward, done, info
    
    def _find_clickable(self, action_arg):
        """查找可点击元素，支持大小写不敏感匹配"""
        if self.text_to_clickable is None:
            self.get_available_actions()
        
        # 精确匹配
        if action_arg in self.text_to_clickable:
            return self.text_to_clickable[action_arg]
        
        # 大小写不敏感匹配
        action_lower = action_arg.lower()
        for key, element in self.text_to_clickable.items():
            if key.lower() == action_lower:
                return element
        
        return None
    
    def highlight_element(self, element, color='red', duration=2):
        """高亮显示元素（在元素周围画框）
        
        Args:
            element: Selenium WebElement
            color: 边框颜色
            duration: 高亮持续时间（秒）
        """
        if element is None:
            return
        
        # 注入 CSS 样式
        highlight_js = f"""
        var element = arguments[0];
        var originalStyle = element.getAttribute('style') || '';
        element.setAttribute('style', originalStyle + '; border: 3px solid {color} !important; box-shadow: 0 0 10px {color} !important;');
        setTimeout(function() {{
            element.setAttribute('style', originalStyle);
        }}, {duration * 1000});
        """
        try:
            self.browser.execute_script(highlight_js, element)
        except Exception:
            pass
    
    def highlight_action(self, action_arg):
        """高亮显示要执行的动作对应的元素
        
        Args:
            action_arg: 动作参数（点击目标）
        
        Returns:
            element: 找到的元素，如果没找到返回 None
        """
        element = self._find_clickable(action_arg)
        if element:
            self.highlight_element(element, color='red', duration=3)
        return element
    
    def highlight_search_bar(self, color='blue', duration=3):
        """高亮显示搜索框
        
        Args:
            color: 边框颜色（默认蓝色）
            duration: 高亮持续时间（秒）
        """
        try:
            search_bar = self.browser.find_element(By.ID, 'search_input')
            self.highlight_element(search_bar, color=color, duration=duration)
            return search_bar
        except Exception:
            return None
    
    def get_available_actions(self):
        """Returns list of available actions at the current step"""
        from selenium.common.exceptions import StaleElementReferenceException
        
        # Determine if a search bar is available
        try:
            search_bar = self.browser.find_element(By.ID, 'search_input')
        except Exception:
            has_search_bar = False
        else:
            has_search_bar = True

        # Collect buttons, links, and options as clickables
        # 添加重试逻辑处理 StaleElementReferenceException
        max_retries = 3
        for attempt in range(max_retries):
            try:
                buttons = self.browser.find_elements(By.CLASS_NAME, 'btn')
                product_links = self.browser.find_elements(By.CLASS_NAME, 'product-link')
                buying_options = self.browser.find_elements(By.CSS_SELECTOR, "input[type='radio']")

                self.text_to_clickable = {}
                for b in buttons + product_links:
                    try:
                        text = b.text
                        if text:  # 只添加非空文本
                            self.text_to_clickable[text] = b
                    except StaleElementReferenceException:
                        continue  # 跳过失效的元素
                
                for opt in buying_options:
                    try:
                        opt_value = opt.get_attribute('value')
                        if opt_value:
                            self.text_to_clickable[opt_value] = opt
                    except StaleElementReferenceException:
                        continue
                
                break  # 成功，退出重试循环
            except StaleElementReferenceException:
                if attempt < max_retries - 1:
                    time.sleep(0.5)  # 等待页面稳定
                    continue
                else:
                    self.text_to_clickable = {}
        
        return dict(
            has_search_bar=has_search_bar,
            clickables=list(self.text_to_clickable.keys()),
        )

    def _parse_html(self, html=None, url=None):
        """
        Returns web request result wrapped in BeautifulSoup object

        Arguments:
        url (`str`): If no url or html is provided, use the current
            observation (HTML) for parsing.
        """
        if html is None:
            if url is not None:
                html = requests.get(url)
            else:
                html = self.state['html']
        html_obj = BeautifulSoup(html, 'html.parser')
        return html_obj

    def get_reward(self):
        """Get reward value at current step of the environment"""
        html_obj = self._parse_html()
        r = html_obj.find(id='reward')
        r = float(r.findChildren("pre")[0].string) if r is not None else 0.0
        return r
    
    def get_instruction_text(self):
        """Get corresponding instruction text for environment current step"""
        html_obj = self._parse_html(self.browser.page_source)
        instruction_text = html_obj.find(id='instruction-text').h4.text
        return instruction_text
    
    def convert_html_to_text(self, html):
        """Strip HTML of tags and add separators to convert observation into simple mode"""
        texts = self._parse_html(html).findAll(text=True)
        visible_texts = filter(tag_visible, texts)
        observation = ' [SEP] '.join(t.strip() for t in visible_texts if t != '\n')
        return observation
    
    @property
    def state(self):
        """
        State that includes all information. The actual observation are
        likely to be a subset or reduced form of the state.
        """
        return dict(
            url=self.browser.current_url,
            html=self.browser.page_source,
            instruction_text=self.instruction_text,
        )
    
    @property
    def observation(self):
        """Compiles state into either the `html` or `text` observation mode"""
        html = self.state['html']
        if self.observation_mode == 'html':
            return html
        elif self.observation_mode == 'text':
            return self.convert_html_to_text(html)
        else:
            raise ValueError(
                f'Observation mode {self.observation_mode} not supported.'
            )

    @property
    def action_space(self):
        # Recommended to use `get_available_actions` instead
        return NotImplementedError

    @property
    def observation_space(self):
        return NotImplementedError

    def reset(self, session=None):
        """Create a new session and reset environment variables
        
        Arguments:
        session (`str` or `int`): Session ID to use. If int, treated as goal index.
        """
        if session is not None:
            self.session = str(session)
        elif self.assigned_session is not None:
            self.session = str(self.assigned_session)
        else:
            self.session = ''.join(random.choices(string.ascii_lowercase, k=5))
        
        init_url = f'{self.server_url}/{self.session}'
        self.browser.get(init_url)

        self.instruction_text = self.get_instruction_text()

        return self.observation, None

    def render(self, mode='human'):
        # TODO: Render observation in terminal or WebShop website
        return NotImplementedError

    def close(self):
        # TODO: When DB used instead of JSONs, tear down DB here
        self.browser.close()
        print('Browser closed.')

def tag_visible(element):
    """Helper method to strip HTML block of extraneous tags"""
    ignore = {'style', 'script', 'head', 'title', 'meta', '[document]'}
    return (
        element.parent.name not in ignore and not isinstance(element, Comment)
    )
