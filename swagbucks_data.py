"""
Swagbucks Platform Data Collection
Contains all the extracted information from Swagbucks website analysis.
"""

# Swagbucks Platform Information
SWAGBUCKS_DATA = {
    "target_url": "https://www.swagbucks.com",
    "login_url": "https://www.swagbucks.com/p/login",
    "surveys_url": "https://www.swagbucks.com/g/paid-surveys",
    "dashboard_url": "https://www.swagbucks.com/dashboard",
    
    "interaction_steps": {
        "login": [
            "Navigate to login page",
            "Fill email field",
            "Fill password field", 
            "Handle CAPTCHA if present",
            "Click login button",
            "Wait for dashboard redirect"
        ],
        "survey_navigation": [
            "Navigate to surveys section",
            "Find available surveys",
            "Select survey to complete",
            "Start survey",
            "Complete questions",
            "Submit responses"
        ]
    },
    
    "element_locators": {
        "login_form": {
            "email_field": [
                'textbox[placeholder*="Email"]',
                'input[placeholder*="Email or Swag Name"]',
                '[bid="118"]',  # Actual element from accessibility tree
                'input[type="email"]',
                '#emailAddress'
            ],
            "password_field": [
                'textbox[placeholder*="Password"]',
                'input[placeholder*="Password"]',
                '[bid="121"]',  # Actual element from accessibility tree
                'input[type="password"]',
                '#password'
            ],
            "login_button": [
                'button:has-text("Sign in")',
                '[bid="132"]',  # Actual sign in button
                'button[type="submit"]',
                'input[type="submit"]',
                '.login-button'
            ],
            "remember_me": [
                'checkbox[name*="Remember"]',
                '[bid="128"]',  # Actual remember me checkbox
                'input[type="checkbox"]'
            ],
            "show_password": [
                'button:has-text("Show password")',
                '[bid="123"]'  # Show password button
            ],
            "forgot_password": [
                'button:has-text("Forgot password?")',
                '[bid="130"]'  # Forgot password button
            ],
            "google_signin": [
                'button:has-text("Continue with Google")',
                '[bid="90"]',  # Google sign in button
                'iframe[src*="google"]'
            ],
            "apple_signin": [
                'button:has-text("Continue with Apple")',
                '[bid="107"]'  # Apple sign in button
            ]
        },
        
        "survey_elements": {
            "survey_list": [
                '.survey-item',
                '.survey-card',
                '.offer-item',
                '[data-survey-id]',
                '.available-survey'
            ],
            "survey_link": [
                'a[href*="survey"]',
                'a[href*="offer"]',
                '.survey-link',
                '.start-survey',
                '.survey-item a'
            ],
            "survey_title": [
                '.survey-title',
                '.survey-name',
                '.title',
                'h3',
                'h4'
            ],
            "survey_reward": [
                '.survey-reward',
                '.points',
                '.sb',
                '.reward',
                '[class*="reward"]',
                '.reward-amount'
            ],
            "survey_time": [
                '.survey-time',
                '.time',
                '.duration',
                '[class*="time"]',
                '.estimated-time'
            ],
            "join_button": [
                'button:has-text("Join Swagbucks")',
                '[bid="525"]',  # Join button from surveys page
                'button:has-text("Continue with Email")',
                '[bid="231"]'  # Continue with Email button
            ],
            "signup_form": [
                'button:has-text("Continue with Google")',
                '[bid="214"]',  # Google signup
                'button:has-text("Continue with Apple")',
                '[bid="228"]'   # Apple signup
            ]
        },
        
        "question_elements": {
            "question_text": [
                '.question-text',
                '.survey-question',
                'h2',
                'h3',
                '.question-title'
            ],
            "radio_buttons": [
                'input[type="radio"]',
                '.radio-option',
                '.single-choice'
            ],
            "checkboxes": [
                'input[type="checkbox"]',
                '.checkbox-option',
                '.multi-choice'
            ],
            "text_input": [
                'input[type="text"]',
                'textarea',
                '.text-input'
            ],
            "dropdown": [
                'select',
                '.dropdown',
                '.select-option'
            ],
            "slider": [
                'input[type="range"]',
                '.slider',
                '.rating-scale'
            ],
            "next_button": [
                'button:has-text("Next")',
                'button:has-text("Continue")',
                'input[type="submit"]',
                '.next-button',
                '.continue-button'
            ]
        }
    },
    
    "waiting_conditions": {
        "page_load": 5000,
        "element_visible": 3000,
        "form_submission": 10000,
        "survey_load": 15000,
        "captcha_solve": 30000
    },
    
    "xpath_selectors": {
        "login_email": "//input[@name='emailAddress' or @id='emailAddress' or @type='email']",
        "login_password": "//input[@name='password' or @id='password' or @type='password']",
        "login_submit": "//button[@type='submit'] | //input[@type='submit'] | //button[contains(text(), 'Log In')]",
        "survey_items": "//div[contains(@class, 'survey')] | //a[contains(@href, 'survey')]",
        "question_text": "//h2 | //h3 | //div[contains(@class, 'question')]",
        "radio_options": "//input[@type='radio']",
        "checkbox_options": "//input[@type='checkbox']",
        "text_inputs": "//input[@type='text'] | //textarea",
        "dropdowns": "//select",
        "next_buttons": "//button[contains(text(), 'Next')] | //button[contains(text(), 'Continue')]"
    },
    
    "css_selectors": {
        "login_form": "#login-form, .login-form, form[action*='login']",
        "email_input": "input[name='emailAddress'], input[type='email'], #emailAddress",
        "password_input": "input[name='password'], input[type='password'], #password",
        "submit_button": "button[type='submit'], input[type='submit'], .login-button",
        "survey_container": ".surveys-container, .survey-list, #surveys",
        "survey_card": ".survey-card, .survey-item, [data-survey-id]",
        "question_container": ".question-container, .survey-question, .question",
        "answer_options": ".answer-option, .choice, input[type='radio'], input[type='checkbox']",
        "text_field": "input[type='text'], textarea, .text-input",
        "dropdown_field": "select, .dropdown, .select-field",
        "continue_button": ".next-button, .continue-button, button:contains('Next')"
    },
    
    "data_points": {
        "user_profile": [
            "age", "gender", "location", "income", "education",
            "employment_status", "household_size", "interests"
        ],
        "survey_metadata": [
            "survey_id", "title", "description", "reward_amount",
            "estimated_time", "category", "provider", "difficulty"
        ],
        "question_types": [
            "single_choice", "multiple_choice", "text_input", 
            "rating_scale", "dropdown", "slider", "date_input",
            "number_input", "email_input", "phone_input"
        ],
        "response_data": [
            "question_id", "question_text", "answer_value",
            "response_time", "question_type", "options"
        ]
    },
    
    "navigation_links": {
        "main_menu": [
            "Dashboard", "Surveys", "Watch", "Shop", "Play",
            "Search", "Discover", "Coupons", "Refer"
        ],
        "survey_categories": [
            "Quick Surveys", "High Paying", "New Surveys",
            "Brand Surveys", "Product Testing", "Focus Groups"
        ],
        "account_links": [
            "Profile", "Settings", "Rewards", "History",
            "Help", "Support", "Logout"
        ]
    },
    
    "dynamic_content_indicators": {
        "loading_spinners": [
            ".loading", ".spinner", ".loader", 
            "[data-loading]", ".progress"
        ],
        "ajax_containers": [
            "[data-ajax]", ".dynamic-content", 
            ".survey-content", ".question-container"
        ],
        "lazy_load_triggers": [
            ".lazy-load", "[data-lazy]", 
            ".infinite-scroll", ".load-more"
        ]
    },
    
    "error_blocking_signals": {
        "captcha_indicators": [
            ".captcha", ".recaptcha", "#captcha",
            "iframe[src*='recaptcha']", ".hcaptcha",
            '[bid="b"]',  # reCAPTCHA iframe found on login page
            'IframePresentational[title="reCAPTCHA"]'
        ],
        "error_messages": [
            ".error", ".alert-error", ".warning",
            ".invalid", ".field-error", ".form-error"
        ],
        "blocking_overlays": [
            ".modal", ".popup", ".overlay",
            ".blocking-message", ".maintenance"
        ],
        "rate_limiting": [
            ".rate-limit", ".too-many-requests",
            ".please-wait", ".cooldown"
        ],
        "account_issues": [
            ".account-suspended", ".verification-required",
            ".login-required", ".access-denied"
        ]
    },
    
    "form_fields_buttons": {
        "input_types": {
            "text": "input[type='text']",
            "email": "input[type='email']", 
            "password": "input[type='password']",
            "number": "input[type='number']",
            "tel": "input[type='tel']",
            "url": "input[type='url']",
            "date": "input[type='date']",
            "radio": "input[type='radio']",
            "checkbox": "input[type='checkbox']",
            "range": "input[type='range']",
            "textarea": "textarea",
            "select": "select"
        },
        "button_types": {
            "submit": "button[type='submit'], input[type='submit']",
            "button": "button[type='button']",
            "reset": "button[type='reset'], input[type='reset']",
            "link_button": "a.button, a.btn",
            "custom_button": ".button, .btn, [role='button']"
        },
        "form_containers": [
            "form", ".form", ".survey-form",
            ".question-form", ".login-form"
        ]
    },
    
    "anti_detection_patterns": {
        "mouse_movements": {
            "natural_curves": True,
            "random_pauses": True,
            "human_like_speed": True
        },
        "typing_patterns": {
            "variable_speed": True,
            "occasional_backspace": True,
            "natural_pauses": True
        },
        "timing_patterns": {
            "question_read_time": (2, 8),
            "answer_think_time": (1, 5),
            "page_transition_time": (1, 3)
        }
    }
}

# Survey Question Types and Patterns
QUESTION_PATTERNS = {
    "demographic": [
        "age", "gender", "income", "education", "location",
        "employment", "household", "marital status"
    ],
    "brand_awareness": [
        "brand recognition", "brand preference", "brand usage",
        "advertising recall", "brand perception"
    ],
    "product_feedback": [
        "product satisfaction", "feature importance", "usage frequency",
        "purchase intent", "recommendation likelihood"
    ],
    "lifestyle": [
        "hobbies", "interests", "activities", "media consumption",
        "shopping habits", "travel preferences"
    ]
}

# Common Survey Response Templates
RESPONSE_TEMPLATES = {
    "age_ranges": ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"],
    "income_ranges": ["Under $25k", "$25k-$50k", "$50k-$75k", "$75k-$100k", "Over $100k"],
    "education_levels": ["High School", "Some College", "Bachelor's", "Master's", "PhD"],
    "employment_status": ["Full-time", "Part-time", "Self-employed", "Student", "Retired", "Unemployed"],
    "frequency_scales": ["Never", "Rarely", "Sometimes", "Often", "Always"],
    "satisfaction_scales": ["Very Dissatisfied", "Dissatisfied", "Neutral", "Satisfied", "Very Satisfied"],
    "likelihood_scales": ["Very Unlikely", "Unlikely", "Neutral", "Likely", "Very Likely"]
}