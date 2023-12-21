from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

def return_yt_comments(url):
    data = []

    options = webdriver.ChromeOptions()
    options.add_argument('--headless')  # Run Chrome in headless mode (without GUI)

    with webdriver.Chrome(options=options) as driver:
        wait = WebDriverWait(driver, 15)
        driver.get(url)

        for item in range(5):
            wait.until(EC.visibility_of_element_located((By.TAG_NAME, "body"))).send_keys(Keys.END)
            time.sleep(2)

        for comment in wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "#content"))):
            data.append(comment.text)

    return data

# Example usage:
youtube_url = 'https://www.youtube.com/somevideo'

comments = return_yt_comments(youtube_url)
print(comments)
