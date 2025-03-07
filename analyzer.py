# analyzer.py
import time
import re
from datetime import datetime
from typing import List, Dict
from collections import Counter
from bs4 import BeautifulSoup
from openai import OpenAI
from firecrawl import FirecrawlApp

class LLMEnhancedAnalyzer:
    def __init__(self, firecrawl_api_key: str, openai_api_key: str):
        self.firecrawl = FirecrawlApp(api_key=firecrawl_api_key)
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.article_intent = ""
        self.secondary_keywords = []

    def set_content_parameters(self, intent: str, keywords: List[str]):
        self.article_intent = intent
        self.secondary_keywords = keywords

    def extract_serp_data(self, data: Dict) -> Dict:
        return {
            'organic_results': self.extract_organic_results(data),
            'paa_questions': self.extract_paa_questions(data),
            'related_searches': self.extract_related_searches(data),
            'search_parameters': data.get('search_parameters', {})  # include search parameters if available
        }

    def extract_organic_results(self, data: Dict) -> List[Dict]:
        results = []
        for article in data.get('organic_results', []):
            result = {
                'title': article.get('title', ''),
                'link': article.get('link', ''),
                'date': article.get('date', ''),
                'snippet': article.get('snippet', ''),
                'position': article.get('position', ''),
                'displayed_link': article.get('displayed_link', '')
            }
            results.append(result)
        return results

    def extract_paa_questions(self, data: Dict) -> List[Dict]:
        questions = []
        for question in data.get('related_questions', []):
            questions.append({
                'question': question.get('question', ''),
                'snippet': question.get('snippet', ''),
                'title': question.get('title', '')
            })
        return questions

    def extract_related_searches(self, data: Dict) -> List[Dict]:
        return [{'query': search.get('query', '')} for search in data.get('related_searches', [])]

    def scrape_competitor_content(self, urls: List[str]) -> List[Dict]:
        scraped_content = []
        for url in urls:
            try:
                params = {'formats': ['markdown', 'html']}
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        result = self.firecrawl.scrape_url(url, params=params)
                        content = result.get('html', result.get('markdown', ''))
                        analysis = self.analyze_content(content)
                        content_data = {
                            'url': url,
                            'content': content,
                            'analysis': analysis
                        }
                        scraped_content.append(content_data)
                        print(f"Successfully scraped competitor URL: {url}")
                        break
                    except Exception as e:
                        if attempt == max_retries - 1:
                            print(f"Error scraping {url}: {str(e)}")
                        else:
                            print(f"Retry {attempt + 1} for {url}")
                            time.sleep(5)
            except Exception as e:
                print(f"Error processing {url}: {str(e)}")
                continue
        return scraped_content

    def scrape_citations(self, urls: List[str]) -> List[Dict]:
        """Scrape content from citation URLs extracted from deep research output."""
        citation_content = []
        for url in urls:
            try:
                params = {'formats': ['markdown', 'html']}
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        result = self.firecrawl.scrape_url(url, params=params)
                        content = result.get('html', result.get('markdown', ''))
                        analysis = self.analyze_content(content)
                        citation_data = {
                            'url': url,
                            'content': content,
                            'analysis': analysis
                        }
                        citation_content.append(citation_data)
                        print(f"Successfully scraped citation URL: {url}")
                        break
                    except Exception as e:
                        if attempt == max_retries - 1:
                            print(f"Error scraping citation {url}: {str(e)}")
                        else:
                            print(f"Retry {attempt + 1} for citation {url}")
                            time.sleep(5)
            except Exception as e:
                print(f"Error processing citation {url}: {str(e)}")
                continue
        return citation_content

    def analyze_content(self, content: str) -> Dict:
        try:
            soup = BeautifulSoup(content, 'html.parser')
            text_content = soup.get_text() if soup.get_text() else content
            analysis = {
                'word_count': len(text_content.split()),
                'common_phrases': self.extract_common_phrases(text_content),
                'content_structure': self.analyze_content_structure(text_content),
                'key_topics': self.extract_key_topics(text_content),
                'content_elements': self.identify_content_elements(content)
            }
            return analysis
        except Exception as e:
            print(f"Error in content analysis: {str(e)}")
            return {}

    def get_llm_analysis(self, context: str, system_prompt: str) -> str:
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",  # or your preferred model
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": context}
                ],
                temperature=0.7,
                max_tokens=3000
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error in LLM analysis: {str(e)}")
            return ""

    def prepare_llm_context(self, scraped_data: List[Dict], serp_data: Dict, deep_thinking: str, citation_data: List[Dict]) -> str:
        serp_analysis = self.extract_serp_data(serp_data)
        citations_formatted = self.format_citation_data(citation_data)
        context = f"""
Search Query: {serp_data.get('search_parameters', {}).get('q', '')}

Content Parameters:
Article Intent: {self.article_intent}
Secondary Keywords: {', '.join(self.secondary_keywords)}

--- Top Ranking Articles ---
{self.format_top_articles(serp_analysis['organic_results'])}

--- People Also Ask Questions ---
{self.format_paa_questions(serp_analysis['paa_questions'])}

--- Related Searches ---
{self.format_related_searches(serp_analysis['related_searches'])}

--- Competitor Content Analysis ---
{self.format_competitor_content(scraped_data)}

--- Deep Research Thinking Process ---
{deep_thinking}

--- Citation Content from Deep Research ---
{citations_formatted}
"""
        return context

    def format_citation_data(self, citation_data: List[Dict]) -> str:
        summaries = []
        for data in citation_data:
            analysis = data.get('analysis', {})
            summary = f"URL: {data.get('url', '')}\nWord Count: {analysis.get('word_count', 0)}\nKey Topics: {', '.join(analysis.get('key_topics', [])[:5])}"
            summaries.append(summary)
        return "\n\n".join(summaries)

    def generate_enhanced_outline(self, serp_data: Dict, scraped_data: List[Dict], deep_thinking: str, citation_data: List[Dict]) -> str:
        print("Preparing context for LLM analysis...")
        context = self.prepare_llm_context(scraped_data, serp_data, deep_thinking, citation_data)
        prompt =  f"""Create a comprehensive SEO article outline for: {serp_data.get('search_parameters', {}).get('q', '')}

Target Audience:
- Primary: {serp_data.get('search_parameters', {}).get('q', '')}
- Secondary: {serp_data.get('search_parameters', {}).get('q', '')}
- Industry level: {serp_data.get('search_parameters', {}).get('q', '')}

SEO Elements to Include:
1. Recommended meta title (50-60 characters)
2. Meta description (130-155 characters)
3. Primary keyword: {serp_data.get('search_parameters', {}).get('q', '')}
   Secondary keywords: {', '.join(self.secondary_keywords)}
4. Search intent: {self.article_intent}
5. Suggested internal linking topics and detailed internal linking methods (e.g., linking to pillar pages, related articles, or product pages)
6. Types of external sources to reference
7. Deep Research Insights: [Insert key insights from the deep research output]
8. If there is a date mentioned in the H1 tag, use only the present year (2025).

Please structure the output exactly as follows:

Primary keyword: [Insert primary keyword]
Secondary keywords: [Insert secondary keywords]

Meta title: [Insert optimized title]
Meta description: [Insert compelling description]

Slug: [Insert primary keyword as slug]

Outline:

H1 Options: [Provide 3-5 title options]

Introduction: [Outline approach and key points]

H2: [Main section title]
  - H3: [Subsection points]
  - H3: [Subsection points]
[Continue with all H2 and H3 sections]

Conclusion: [Outline approach]

FAQ:
1. [Question 1]
2. [Question 2]
3. [Question 3]
4. [Question 4]
5. [Question 5]

Writing Guidelines:
- Word count target: [Predict based on competitor analysis]
- Content tone: Professional
- Statistics/data placement
- Expert quote areas
- Visual content opportunities
- Content upgrades/lead magnets
- Key takeaways
- Internal/external linking strategy

Article Type Prediction:

Based on SERP analysis, competitor data, deep research insights, and {serp_data.get('search_parameters', {}).get('q', '')}, the best article format for this topic is:
[Insert predicted article type - e.g., "How-To Guide," "Listicle," "Comparison Blog," "Technical Article," "Product Review," etc.]

Justification:
- [Explain why this format is ideal based on user search behavior, top-ranking content structures, competitor trends, and deep research insights]
"""

        llm_outline = self.get_llm_analysis(context, prompt)
        return self.format_llm_outline(llm_outline, serp_data)

    def format_llm_outline(self, llm_output: str, serp_data: Dict) -> str:
        try:
            return f"""SEO Article Outline for: "{serp_data.get('search_parameters', {}).get('q', '')}"

{llm_output}

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        except Exception as e:
            print(f"Error formatting LLM outline: {str(e)}")
            return "Error generating outline"

    def generate_research_report(self, deep_thinking: str, scraped_data: List[Dict]) -> str:
        """Generate a separate research report showing the deep research thinking and competitor analysis summary."""
        competitor_summary = self.format_competitor_content(scraped_data)
        report = f"""=== Research Report ===

Deep Research Thinking Process:
{deep_thinking}

--- Competitor Content Analysis Summary ---
{competitor_summary}

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        return report

    # Helper formatting methods
    def format_top_articles(self, results: List[Dict]) -> str:
        try:
            return "\n".join([f"- {result['title']} (URL: {result['link']})" for result in results[:5]])
        except Exception as e:
            print(f"Error formatting top articles: {str(e)}")
            return ""

    def format_paa_questions(self, questions: List[Dict]) -> str:
        try:
            return "\n".join([f"- {q['question']}" for q in questions])
        except Exception as e:
            print(f"Error formatting PAA questions: {str(e)}")
            return ""

    def format_related_searches(self, searches: List[Dict]) -> str:
        try:
            return "\n".join([f"- {search['query']}" for search in searches])
        except Exception as e:
            print(f"Error formatting related searches: {str(e)}")
            return ""

    def format_competitor_content(self, scraped_data: List[Dict]) -> str:
        try:
            summaries = []
            for data in scraped_data:
                analysis = data.get('analysis', {})
                summary = f"URL: {data.get('url', '')}\nWord Count: {analysis.get('word_count', 0)}\nKey Topics: {', '.join(analysis.get('key_topics', [])[:5])}"
                summaries.append(summary)
            return "\n\n".join(summaries)
        except Exception as e:
            print(f"Error formatting competitor content: {str(e)}")
            return ""

    def extract_common_phrases(self, text_content: str) -> List[str]:
        try:
            phrases = re.findall(r'\b[\w\s]{10,30}\b', text_content.lower())
            phrase_counter = Counter(phrases)
            return [phrase for phrase, count in phrase_counter.most_common(10)]
        except Exception as e:
            print(f"Error extracting common phrases: {str(e)}")
            return []

    def analyze_content_structure(self, text_content: str) -> Dict:
        try:
            paragraphs = text_content.split('\n\n')
            structure = {
                'total_paragraphs': len(paragraphs),
                'avg_paragraph_length': sum(len(p.split()) for p in paragraphs) / len(paragraphs) if paragraphs else 0,
            }
            return structure
        except Exception as e:
            print(f"Error analyzing content structure: {str(e)}")
            return {}

    def extract_key_topics(self, text_content: str) -> List[str]:
        try:
            words = re.findall(r'\b\w+\b', text_content.lower())
            word_counter = Counter(words)
            return [word for word, count in word_counter.most_common(10)]
        except Exception as e:
            print(f"Error extracting key topics: {str(e)}")
            return []

    def identify_content_elements(self, content: str) -> Dict:
        try:
            soup = BeautifulSoup(content, 'html.parser')
            elements = {
                'lists': len(soup.find_all(['ul', 'ol'])),
                'tables': len(soup.find_all('table')),
                'images': len(soup.find_all('img')),
                'links': len(soup.find_all('a')),
                'headings': len(soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']))
            }
            return elements
        except Exception as e:
            print(f"Error identifying content elements: {str(e)}")
            return {}
