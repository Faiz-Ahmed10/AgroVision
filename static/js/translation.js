document.addEventListener('DOMContentLoaded', function() {
    console.log("Translation script loaded");
    
    // Create language selector with Tailwind classes to match your design
    const languageSelector = `
        <select id="languageSelector" class="bg-white/10 text-green-400 border border-green-400 rounded px-2 py-1 ml-4 hover:border-green-300 hover:text-green-300 transition-colors">
            <option value="en">English</option>
            <option value="hindi">हिंदी</option>
            <option value="urdu">اردو</option>
            <option value="telugu">తెలుగు</option>
        </select>
    `;
    
    // Try to find the navbar menu div
    const navbarMenu = document.querySelector('.flex.space-x-8');
    if (navbarMenu) {
        // If navbar exists (feature pages), append to the menu
        navbarMenu.insertAdjacentHTML('beforeend', languageSelector);
    } else {
        // If no navbar (index page), create a floating language selector
        const floatingSelector = document.createElement('div');
        floatingSelector.className = 'fixed top-4 right-4 z-20';
        floatingSelector.innerHTML = languageSelector;
        document.body.appendChild(floatingSelector);
    }

    // Rest of your translation code remains the same
    async function translateContent(text, targetLang) {
        console.log(`Attempting to translate: ${text} to ${targetLang}`);
        try {
            const response = await fetch('/translate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    text: text,
                    language: targetLang
                })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            console.log(`Translation successful: ${data.translated_text}`);
            return data.translated_text;
        } catch (error) {
            console.error('Translation error:', error);
            return text;
        }
    }

    async function translatePage(language) {
        const elementsToTranslate = document.querySelectorAll('[data-translate]');
        console.log(`Found ${elementsToTranslate.length} elements to translate`);
        
        for (const element of elementsToTranslate) {
            const originalText = element.getAttribute('data-translate');
            console.log(`Translating element: ${originalText}`);
            const translatedText = await translateContent(originalText, language);
            element.textContent = translatedText;
        }
    }

    // Add event listener to language selector
    const selector = document.getElementById('languageSelector');
    if (selector) {
        console.log("Language selector found");
        selector.addEventListener('change', function(e) {
            console.log(`Language changed to: ${e.target.value}`);
            const selectedLanguage = e.target.value;
            if (selectedLanguage !== 'en') {
                translatePage(selectedLanguage);
            } else {
                // Restore original English content
                document.querySelectorAll('[data-translate]').forEach(element => {
                    element.textContent = element.getAttribute('data-translate');
                });
            }
        });
    }
});