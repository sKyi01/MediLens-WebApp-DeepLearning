from main import app
import unittest

class TestToPerform(unittest.TestCase):
    def settup(self):
        pass
        self.app=app.test_client()
        
    def test_page(self):
        response0=self.app.get("/",follow_redirects=True)
        print(response0)
        self.assertEqual(response0.status_code,200)
        
    def test_page_disease(self):
        response1=self.app.get("/disease",follow_redirects=True)
        print(response1)
        self.assertEqual(response1.status_code,200)
        
    
    
    
        
    def test_page_heartdisease(self):
        response4=self.app.get("/heart",follow_redirects=True)
        print(response4)
        self.assertEqual(response4.status_code,200)
        
    def test_page_pneumonia(self):
        response5=self.app.get("/pneumonia",follow_redirects=True)
        print(response5)
        self.assertEqual(response5.status_code,200)
    
    
    
if __name__=="__main__":
    unittest.main()