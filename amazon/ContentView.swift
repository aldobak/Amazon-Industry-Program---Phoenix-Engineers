//
//  ContentView.swift
//  amazon
//
//  Created by Aldo on 11/08/2024.
//

import SwiftUI
// Struct to represent user input
struct RecommendationInput: Codable{
    var roi: Double
    var maxInvestment:Double
    var period:Double
}
// URL endpoint that the client will use to contact the server
class RecommendationData: ObservableObject {
    @Published var recommendations: [String] = []
}
// Function sending a post request to the endpoint with user input
func sendData(data: RecommendationInput, recommendationData: RecommendationData) {
    do {
        let url = URL(string: "http://10.169.11.31:5000/getRecommendation")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        let sentData = try JSONEncoder().encode(data)
        request.httpBody = sentData

        let task = URLSession.shared.dataTask(with: request) { (responseData, response, error) in
            if let error = error {
                print("Error: \(error.localizedDescription)")
                return
            }

            guard let data = responseData else {
                print("No data received")
                return
            }

            // Log raw response data for debugging
            if let jsonString = String(data: data, encoding: .utf8) {
                print("Raw JSON Response: \(jsonString)")
            } else {
                print("Failed to convert data to String")
            }

            do {
                // Assuming the response is a JSON array
                let decodedList = try JSONDecoder().decode([String].self, from: data)
                print("Decoded List: \(decodedList)")
                DispatchQueue.main.async {
                    recommendationData.recommendations = decodedList
                }
            } catch {
                print("Decoding error: \(error.localizedDescription)")
            }
        }

        task.resume()
    } catch {
        print("Encoding error: \(error.localizedDescription)")
    }
}


func display(){
    let url = URL(string:"http://10.169.11.31:5000/display")!
    var request = URLRequest(url:url)
    request.httpMethod = "GET"
    let task = URLSession.shared.dataTask(with: request)
    task.resume()
    print(request)
}

class UserSession: ObservableObject {
    @Published var isLoggedIn: Bool = false
    @Published var username: String = ""
    @Published var password: String = ""
}

struct LoginView: View {
    @EnvironmentObject var session: UserSession
    @State private var showError: Bool = false
    
    // Predefined credentials
    let correctUsername = "Aldo"
    let correctPassword = "123"
    
    var body: some View {
        VStack {
            Spacer()
            
            // Spartan Helmet Logo
            Image("aureliuslogo")
                .resizable()
                .scaledToFit()
                .frame(width: 350.0, height: 120)
                .padding(.bottom, 20)
            
            // Login Title
            Text("Login")
                .font(.largeTitle)
                .bold()
                .padding(.bottom, 40)
            
            // Email Field
            VStack(alignment: .leading) {
                Text("Email")
                    .font(.headline)
                
                TextField("Email", text: $session.username)
                    .padding()
                    .background(Color(.systemGray6))
                    .cornerRadius(10)
                    .autocapitalization(.none)
                    .disableAutocorrection(true)
            }
            .padding(.horizontal, 40)
            
            // Password Field
            VStack(alignment: .leading) {
                Text("Password")
                    .font(.headline)
                
                SecureField("Password", text: $session.password)
                    .padding()
                    .background(Color(.systemGray6))
                    .cornerRadius(10)
                    .autocapitalization(.none)
                    .disableAutocorrection(true)
            }
            .padding(.horizontal, 40)
            .padding(.top, 10)
            
            // Forgot Password Link
            HStack {
                Spacer()
                Button(action: {
                    // Handle forgot password action
                }) {
                    Text("Forgot password?")
                        .font(.subheadline)
                        .foregroundColor(.black)
                }
                .padding(.horizontal, 40)
                .padding(.top, 10)
            }
            
            // Error Message
            if showError {
                Text("Invalid username or password")
                    .foregroundColor(.red)
                    .font(.footnote)
                    .padding(.top, 5)
            }
            
            // Login Button
            Button(action: {
                // Trim whitespace from input
                let trimmedUsername = session.username.trimmingCharacters(in: .whitespaces)
                let trimmedPassword = session.password.trimmingCharacters(in: .whitespaces)
                
                // Validation with predefined credentials
                if trimmedUsername == correctUsername && trimmedPassword == correctPassword {
                    withAnimation {
                        session.isLoggedIn = true
                    }
                } else {
                    withAnimation {
                        showError = true
                    }
                }
            }) {
                Text("Log in")
                    .foregroundColor(.white)
                    .frame(minWidth: 0, maxWidth: .infinity)
                    .padding()
                    .background(Color.black)
                    .cornerRadius(25)
                    .padding(.horizontal, 40)
            }
            .padding(.top, 30)
            
            // Create an Account Link
            HStack {
                Text("Not on Aurelius?")
                    .font(.subheadline)
                
                NavigationLink(destination: CreateAccountView().environmentObject(session)) {
                    Text("Create an account")
                        .font(.subheadline)
                        .foregroundColor(.black)
                        .underline()
                }
            }
            .padding(.top, 10)
            
            Spacer()
        }
        .navigationBarHidden(true)
        .background(Color(UIColor.systemBackground))
        .onAppear {
            // Reset error message when view appears
            showError =
            false
        }
    }
}

struct CreateAccountView: View {
    @EnvironmentObject var session: UserSession
    
    @State private var firstName: String = ""
    @State private var lastName: String = ""
    @State private var retypePassword: String = ""
    
    var body: some View {
        VStack {
            Spacer()
            
            // Spartan Helmet Logo
            Image("aureliuslogo")
                .resizable()
                .scaledToFit()
                .frame(width: 200, height: 125)
                .padding(.bottom, 20)
            
            // Create Account Title
            Text("Create an Account")
                .font(.largeTitle)
                .bold()
                .padding(.bottom, 20)
            
            // First Name Field
            TextField("First Name", text: $firstName)
                .padding()
                .background(Color(.systemGray6))
                .cornerRadius(10)
                .autocapitalization(.words)
                .disableAutocorrection(true)
                .padding(.horizontal, 40)
            
            // Last Name Field
            TextField("Last Name", text: $lastName)
                .padding()
                .background(Color(.systemGray6))
                .cornerRadius(10)
                .autocapitalization(.words)
                .disableAutocorrection(true)
                .padding(.horizontal, 40)
            
            // Email Field
            TextField("Email", text: $session.username)
                .padding()
                .background(Color(.systemGray6))
                .cornerRadius(10)
                .autocapitalization(.none)
                .disableAutocorrection(true)
                .padding(.horizontal, 40)
            
            // Password Field
            SecureField("Password", text: $session.password)
                .padding()
                .background(Color(.systemGray6))
                .cornerRadius(10)
                .autocapitalization(.none)
                .disableAutocorrection(true)
                .padding(.horizontal, 40)
            
            // Retype Password Field
            SecureField("Retype Password", text: $retypePassword)
                .padding()
                .background(Color(.systemGray6))
                .cornerRadius(10)
                .autocapitalization(.none)
                .disableAutocorrection(true)
                .padding(.horizontal, 40)
            
            // Password Requirements
            VStack(alignment: .leading, spacing: 5) {
                Text("Password must:")
                    .font(.footnote)
                Text("• Be at least 10 characters long")
                    .font(.footnote)
                Text("• Not contain first or last name")
                    .font(.footnote)
                Text("• Contain at least 1 special character and 1 number")
                    .font(.footnote)
                Text("• Contain 1 lowercase and 1 uppercase character")
                    .font(.footnote)
            }
            .padding(.horizontal, 40)
            .padding(.top, 10)
            
            // Continue Button
            Button(action: {
                // Handle the account creation process here
            }) {
                Text("Continue")
                    .foregroundColor(.white)
                    .frame(minWidth: 0, maxWidth: .infinity)
                    .padding()
                    .background(Color.black)
                    .cornerRadius(25)
                    .padding(.horizontal, 40)
            }
            .padding(.top, 20)
            
            // Log in Link
            HStack {
                Text("Already on Aurelius?")
                    .font(.subheadline)
                
                NavigationLink(destination: LoginView().environmentObject(session)) {
                    Text("Log in")
                        .font(.subheadline)
                        .foregroundColor(.black)
                        .underline()
                }
            }
            .padding(.top, 10)
            
            Spacer()
        }
        .navigationBarHidden(true)
        .background(Color(UIColor.systemBackground))
    }
}

// Main View
struct ContentView: View {
    @StateObject private var session = UserSession()
    @StateObject private var recommendationData = RecommendationData() // Create the shared object

    var body: some View {
        NavigationView {
            Group {
                if session.isLoggedIn {
                    mainAppView
                        .transition(.opacity)
                        .environmentObject(recommendationData) // Inject shared object into environment
                } else {
                    LoginView()
                        .environmentObject(session)
                        .transition(.opacity)
                }
            }
            .animation(.easeInOut(duration: 0.5), value: session.isLoggedIn)
        }
    }
    
    private var mainAppView: some View {
        TabView {
            HomeView()
                .tabItem {
                    Label("Home", systemImage: "house.fill")
                }
            
            AureliusView()
                .tabItem {
                    Label("Aurelius", systemImage: "star.fill")
                }
            
            BrowseView()
                .tabItem {
                    Label("Browse", systemImage: "magnifyingglass")
                }
            
            ProfileView()
                .tabItem {
                    Label("Profile", systemImage: "person.fill")
                }
        }
    }
}

// Home View
struct HomeView: View {
    @State private var isNewsExpanded: Bool = false
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    // Header Section
                    HeaderView()
                    
                    // Stock Tiles Section
                    StockTilesView()
                    
                    // News Section
                    NewsSectionView(isNewsExpanded: $isNewsExpanded)
                }
                .navigationBarHidden(true)
                .padding()
            }
            .fullScreenCover(isPresented: $isNewsExpanded) {
                ExpandedNewsView(isNewsExpanded: $isNewsExpanded)
            }
        }
    }
}

// Header View
struct HeaderView: View {
    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Welcome, Marcus")
                .font(.largeTitle)
                .bold()
            
            Text("Total Asset")
                .font(.title3)
                .foregroundColor(.gray)
            
            ZStack {
                RoundedRectangle(cornerRadius: 15)
                    .fill(Color.black.opacity(0.9))
                    .frame(height: 120)
                
                VStack(alignment: .leading, spacing: 10) {
                    Text("Your total asset portfolio")
                        .font(.title2)
                        .foregroundColor(.white)
                    
                    HStack {
                        Text("$2.240.559")
                            .font(.largeTitle)
                            .bold()
                            .foregroundColor(.white)
                        
                        Spacer()
                        
                        HStack(spacing: 5) {
                            Image(systemName: "arrow.up.right")
                                .foregroundColor(.green)
                            Text("+2%")
                                .foregroundColor(.green)
                        }
                    }
                }
                .padding()
            }
        }
    }
}

// Stock Tiles View
struct StockTilesView: View {
    let stocks = [
        ("APPL", "5.7%", Image("applelogo")),
        ("NFLX", "7.5%", Image("netflixlogo")),
        ("AMZN", "2.1%", Image("amazonlogo")),
        ("TSLA", "1.46%", Image("teslalogo"))
    ]
    
    var body: some View {
        VStack(alignment: .leading) {
            Text("Ask Aurelius")
                .font(.title3)
                .bold()
            
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 20) {
                    ForEach(stocks, id: \.0) { stock in
                        VStack {
                            stock.2
                                .resizable()
                                .frame(width: 35, height: 35)
                            Text(stock.0)
                                .font(.caption)
                                .bold()
                            Text(stock.1)
                                .font(.caption2)
                                .foregroundColor(.green)
                        }
                        .frame(width: 80)
                    }
                }
            }
        }
    }
}

// News Section View
struct NewsSectionView: View {
    @Binding var isNewsExpanded: Bool
    
    let newsArticles = [
        ("Record highs amid tech stocks rally", "Strong earnings from Apple", "profile1"),
        ("How To Pick for a Blue Chip", "Oil prices reach five-year high, boosting stocks of ExxonMobil", "profile2"),
        ("Tesla's new model surges", "High demand expected", "profile3"),
        ("Amazon Prime Day results", "Record-breaking sales", "profile4"),
        ("Stock Market Analysis", "What experts are saying", "profile5")
    ]
    
    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text("What's new")
                    .font(.title3)
                    .foregroundColor(.pink)
                    .bold()
                
                Spacer()
                
                Button(action: {
                    isNewsExpanded = true
                }) {
                    Text("See All →")
                        .foregroundColor(.pink)
                        .font(.callout)
                }
            }
            
            VStack(alignment: .leading, spacing: 15) {
                ForEach(0..<2) { index in
                    NewsItemView(title: newsArticles[index].0, subtitle: newsArticles[index].1, imageName: newsArticles[index].2)
                    
                    if index < 1 { // Add divider only between items, not after the last one
                        Divider()
                            .background(Color.white)
                    }
                }
            }
            .padding()
            .background(Color.black.opacity(0.9))
            .cornerRadius(15)
        }
    }
}

// Expanded News View (Full-Screen)
struct ExpandedNewsView: View {
    @Binding var isNewsExpanded: Bool
    
    let newsArticles = [
        ("Record highs amid tech stocks rally", "Strong earnings from Apple", "profile1"),
        ("How To Pick for a Blue Chip", "Oil prices reach five-year high, boosting stocks of ExxonMobil", "profile2"),
        ("Tesla's new model surges", "High demand expected", "profile3"),
        ("Amazon Prime Day results", "Record-breaking sales", "profile4"),
        ("Stock Market Analysis", "What experts are saying", "profile5")
    ]
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(alignment: .leading, spacing: 15) {
                    ForEach(Array(newsArticles.enumerated()), id: \.element.0) { index, article in
                        NewsItemView(title: article.0, subtitle: article.1, imageName: article.2)
                        
                        if index < newsArticles.count - 1 {
                            Divider()
                                .background(Color.white)
                        }
                    }
                }
                .padding()
                .background(Color.black.opacity(0.9))
                .cornerRadius(15)
                .padding()
            }
            .navigationTitle("News")
            .navigationBarItems(trailing: Button("Close") {
                isNewsExpanded = false
            })
        }
    }
}

// News Item View
struct NewsItemView: View {
    let title: String
    let subtitle: String
    let imageName: String
    
    var body: some View {
        HStack {
            VStack(alignment: .leading) {
                Text(title)
                    .font(.headline)
                    .foregroundColor(.white)
                
                Text(subtitle)
                    .font(.subheadline)
                    .foregroundColor(.white)
                    .opacity(0.7)
            }
            
            Spacer()
            
            Image(imageName)
                .resizable()
                .frame(width: 40, height: 40)
                .clipShape(Circle())
        }
    }
}

// Aurelius View
struct AureliusView: View {
    @State private var desiredRiskMargin: Double = 50
    @State private var desiredROI: Double = 50
    @State private var maximumInvestmentAmount: Double = 5000
    @State private var investmentTimeFrame: Double = 30
    @State private var additionalInfo: String = ""
    @State private var showRecommendationPage: Bool = false
    @EnvironmentObject var recommendationData: RecommendationData // Access the shared object

    var body: some View {
        NavigationView {
            VStack(spacing: 0) {
                headerContent
                tabContent
                Spacer(minLength: 20)
            }
            .padding()
            .navigationTitle("")
            .navigationBarHidden(true)
            .background(
                NavigationLink(destination: RecommendationView(), isActive: $showRecommendationPage) {
                    EmptyView()
                }
            )
        }
    }

    var headerContent: some View {
        VStack {
            Image("aureliuslogo")
                .resizable()
                .scaledToFit()
                .frame(width: 100, height: 100)
                .padding(.bottom, 1)
            
            Text("Aurelius")
                .font(.largeTitle)
                .bold()
            
            Text("Let Aurelius help you choose your next investment.")
                .fixedSize(horizontal: false, vertical: true)
                .multilineTextAlignment(.center)
                .padding(.bottom, 10)
        }
    }

    var tabContent: some View {
        TabView {
            VStack {
                sliderView(label: "What ROI are you aiming for?", value: $desiredROI, unit: "%", range: 0...100)
                sliderView(label: "What's your maximum investment?", value: $maximumInvestmentAmount, unit: "$", range: 100...10000, step: 100)
            }
            .padding(.bottom, 20)
            
            VStack(alignment: .leading, spacing: 10) {
                sliderView(label: "Over how many days do you want to invest?", value: $investmentTimeFrame, unit: "days", range: 30...120, step: 30)
                additionalInfoView
                recommendButton
            }
            .padding(.bottom, 20)
        }
        .tabViewStyle(PageTabViewStyle(indexDisplayMode: .always))
        .indexViewStyle(PageIndexViewStyle(backgroundDisplayMode: .always))
        .frame(height: 450)
        .padding(.bottom, 30)
    }

    func sliderView(label: String, value: Binding<Double>, unit: String, range: ClosedRange<Double>, step: Double = 1) -> some View {
        VStack(alignment: .leading) {
            Text(label)
                .font(.headline)
            Slider(value: value, in: range, step: step)
            Text("\(Int(value.wrappedValue)) \(unit)")
                .bold()
        }
        .padding([.horizontal, .top])
    }

    var additionalInfoView: some View {
        VStack(alignment: .leading) {
            Text("Anything else we should know?")
                .font(.headline)
            TextField("Enter details...", text: $additionalInfo)
                .textFieldStyle(RoundedBorderTextFieldStyle())
                .padding(.bottom, 10)
        }
        .padding([.horizontal, .top])
    }

    var recommendButton: some View {
        Button(action: {
            let data = RecommendationInput(roi: desiredROI, maxInvestment: maximumInvestmentAmount, period: investmentTimeFrame)
            sendData(data: data, recommendationData: recommendationData) // Pass the shared object directly
            showRecommendationPage = true
        }) {
            Text("Recommend")
                .foregroundColor(.white)
                .frame(minWidth: 0, maxWidth: .infinity)
                .padding()
                .background(Color.blue)
                .cornerRadius(10)
        }
        .padding([.horizontal, .top])
    }

}



// Recommendation View
struct RecommendationView: View {
    @State private var recommendations: [String] = [] // State variable to hold the response
    @EnvironmentObject var recommendationData: RecommendationData // Correctly use @EnvironmentObject

    var body: some View {
        VStack {
            Text("Aurelius")
                .font(.largeTitle)
                .bold()
                .padding(.top, 1)
            
            Image("aureliuslogo")
                .resizable()
                .scaledToFit()
                .frame(width: 100, height: 100)
                .padding(.top, 3)
            
            // Displaying recommendations in a Text view
            Text("Based On Your Inputs...\n\n" + recommendationData.recommendations.joined(separator: "\n"))
                .font(.title2)
                .padding(.top, 7)
            
            Spacer()
        }
        .padding()
        .onAppear {
            
        }
    }

    // Function to call sendData a
}


// Main Profile View
struct ProfileView: View {
    var body: some View {
        NavigationView { // Embed the ProfileView in a NavigationView
            VStack(alignment: .leading, spacing: 0) {
                // Top Bar with Profile Title
                HStack {
                    Text("Profile")
                        .font(.largeTitle)   // Large Title Font
                        .bold()              // Bold Text
                        .padding(.leading)
                    Spacer()
                }
                .padding(.top, 10)
                
                // User Profile Header
                VStack(alignment: .center, spacing: 10) {
                    Image(systemName: "person.crop.circle.fill")
                        .font(.system(size: 80))
                        .foregroundColor(.gray)
                    Text("Marcus Aurelius")
                        .font(.title2)
                        .bold()
                    Text("marcus@gmail.com")
                        .font(.subheadline)
                        .foregroundColor(.gray)
                }
                .padding(.top, 20)
                .frame(maxWidth: .infinity)
                .background(Color.white) // White background for the profile section
                
                // Menu Options with Enhanced Styling
                VStack(alignment: .leading, spacing: 0) {
                    NavigationLink(destination: AccountView()) {
                        HStack {
                            Label("Account", systemImage: "person.circle")
                                .foregroundColor(.black)
                            Spacer()
                            Image(systemName: "chevron.right")
                                .foregroundColor(.gray)
                        }
                        .padding(.horizontal)
                        .padding(.vertical, 12)
                    }
                    Divider()
                    NavigationLink(destination: AssetsView()) {
                        HStack {
                            Label("My Assets", systemImage: "dollarsign.circle")
                                .foregroundColor(.black)
                            Spacer()
                            Image(systemName: "chevron.right")
                                .foregroundColor(.gray)
                        }
                        .padding(.horizontal)
                        .padding(.vertical, 12)
                    }
                    Divider()
                    NavigationLink(destination: SettingsView()) {
                        HStack {
                            Label("Settings", systemImage: "gearshape")
                                .foregroundColor(.black)
                            Spacer()
                            Image(systemName: "chevron.right")
                                .foregroundColor(.gray)
                        }
                        .padding(.horizontal)
                        .padding(.vertical, 12)
                    }
                }
                .background(Color.white)
                .cornerRadius(10)
                .padding(.horizontal)
                
                // Add spacing between Settings and History section
                Spacer(minLength: 20)
                
                // Scrollable History Section
                ScrollView {
                    VStack(alignment: .leading, spacing: 0) {
                        HStack {
                            Text("History")
                                .font(.title3)  // Slightly bigger font for the title
                                .bold()
                            Spacer()
                            NavigationLink(destination: FullHistoryView()) {
                                Text("See All")
                                    .font(.subheadline)
                                    .foregroundColor(.blue)
                            }
                        }
                        .padding([.top, .horizontal])
                        
                        VStack(alignment: .leading, spacing: 10) { // Reduced spacing between history items
                            HistoryItemView(amount: "USD 200.000", action: "Buy \"APPL\" Stock", date: "TUE 22 Jul 2024")
                            Divider()
                            HistoryItemView(amount: "USD 150.000", action: "Sell \"TLKM\" Stock", date: "TUE 22 Jul 2024", amountColor: .green)
                            Divider()
                            HistoryItemView(amount: "USD 1,000.24", action: "Buy \"FB\" Stock", date: "TUE 22 Jul 2024")
                            Divider()
                            HistoryItemView(amount: "USD 500.000", action: "Sell \"GOOG\" Stock", date: "WED 23 Jul 2024", amountColor: .green)
                        }
                        .padding(.horizontal)
                    }
                    .background(Color.white)
                    .cornerRadius(10)
                    .padding(.horizontal)
                }
                
                Spacer()
            }
            .navigationBarHidden(true)
            .background(Color.white.edgesIgnoringSafeArea(.all)) // White background for the entire view
        }
    }
}

// Full History View
struct FullHistoryView: View {
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 0) {
                // History Items
                HistoryItemView(amount: "USD 200.000", action: "Buy \"APPL\" Stock", date: "TUE 22 Jul 2024")
                Divider()
                HistoryItemView(amount: "USD 150.000", action: "Sell \"TLKM\" Stock", date: "TUE 22 Jul 2024", amountColor: .green)
                Divider()
                HistoryItemView(amount: "USD 1,000.24", action: "Buy \"FB\" Stock", date: "TUE 22 Jul 2024")
                Divider()
                HistoryItemView(amount: "USD 500.000", action: "Sell \"GOOG\" Stock", date: "WED 23 Jul 2024", amountColor: .green)
                Divider()
                HistoryItemView(amount: "USD 300.000", action: "Buy \"MSFT\" Stock", date: "THU 24 Jul 2024")
                Divider()
                HistoryItemView(amount: "USD 50.000", action: "Dividend Payment from \"AAPL\"", date: "FRI 25 Jul 2024", amountColor: .green)
                Divider()
                HistoryItemView(amount: "USD 450.000", action: "Sell \"NFLX\" Stock", date: "FRI 25 Jul 2024", amountColor: .green)
                Divider()
                HistoryItemView(amount: "USD 600.000", action: "Buy \"TSLA\" Stock", date: "MON 28 Jul 2024")
                Divider()
                HistoryItemView(amount: "USD 75.000", action: "Dividend Payment from \"GOOG\"", date: "TUE 29 Jul 2024", amountColor: .green)
                Divider()
                HistoryItemView(amount: "USD 100.000", action: "Sell \"AMZN\" Stock", date: "WED 30 Jul 2024", amountColor: .green)
                Divider()
                HistoryItemView(amount: "USD 50.000", action: "Buy \"BABA\" Stock", date: "THU 31 Jul 2024")
                Divider()
                HistoryItemView(amount: "USD 200.000", action: "Sell \"NVDA\" Stock", date: "FRI 01 Aug 2024", amountColor: .green)
                Divider()
                HistoryItemView(amount: "USD 10.000", action: "Dividend Payment from \"MSFT\"", date: "MON 04 Aug 2024", amountColor: .green)
                Divider()
                HistoryItemView(amount: "USD 300.000", action: "Buy \"INTC\" Stock", date: "TUE 05 Aug 2024")
                Divider()
                HistoryItemView(amount: "USD 400.000", action: "Sell \"FB\" Stock", date: "WED 06 Aug 2024", amountColor: .green)
            }
            .padding()
        }
        .navigationTitle("Full History") // Title for the view
    }
}

// History Item View
struct HistoryItemView: View {
    var amount: String
    var action: String
    var date: String
    var amountColor: Color = .black
    
    var body: some View {
        HStack {
            VStack(alignment: .leading) {
                Text(amount)
                    .font(.subheadline)
                    .foregroundColor(amountColor)
                Text(action)
                    .font(.caption)
                    .foregroundColor(.gray)
            }
            Spacer()
            Text(date)
                .font(.caption)
                .foregroundColor(.gray)
        }
        .padding(.vertical, 4) // Reduced vertical padding
    }
}

// Placeholder Views for Navigation Links
struct AccountView: View {
    var body: some View {
        Text("Account Details")
            .font(.title)
            .padding()
    }
}

// Assets View with My Assets Section and Performance Graph
struct AssetsView: View {
    var body: some View {
        VStack(alignment: .leading) {
            // My Assets Title Row
            HStack {
                Text("My Assets")
                    .font(.system(size: 34, weight: .bold))
                    .foregroundColor(.black)
                Spacer()
                Text("$2,240.559")
                    .font(.system(size: 28, weight: .bold))
                    .foregroundColor(.black)
            }
            .padding([.top, .horizontal])
            
            // Daily % Chg. Title Row
            HStack {
                Spacer() // Keeps it aligned to the right side
                Text("Daily % Chg.")
                    .font(.subheadline)
                    .foregroundColor(.gray)
                    .padding(.trailing, 10)
            }
            .padding(.horizontal)
            .padding(.bottom, 5)
            
            List {
                AssetItemView(icon: "applelogo", name: "Apple Inc.", change: "-0.63%", changeColor: .red)
                AssetItemView(icon: "teslalogo", name: "Tesla, Inc.", change: "1.57%", changeColor: .green)
                AssetItemView(icon: "nikelogo", name: "Nike Inc.", change: "1.34%", changeColor: .green)
                AssetItemView(icon: "metalogo", name: "Meta Inc.", change: "-1.63%", changeColor: .red)
                AssetItemView(icon: "nvidialogo", name: "NVIDIA Corporation", change: "5.7%", changeColor: .green)
            }
            .listStyle(PlainListStyle())
            .frame(height: 300) // Adjust to fit content
            
            Text("Performance")
                .font(.system(size: 24))
                .bold()
                .padding([.top, .horizontal])
            
            // Performance Graph
            PerformanceGraphView(dataPoints: [10, 50, 20, 70, 40, 90, 150])
                .frame(height: 200)
                .padding([.leading, .trailing, .bottom])
        }
    }
}

// Asset Item View updated to match the design
struct AssetItemView: View {
    var icon: String
    var name: String
    var change: String
    var changeColor: Color
    
    var body: some View {
        HStack {
            Image(icon) // Use the image from your assets
                .resizable()
                .frame(width: 30, height: 30)
                .padding(.trailing, 10)
            VStack(alignment: .leading) {
                Text(name)
                    .font(.headline)
            }
            Spacer()
            Text(change)
                .font(.headline)
                .foregroundColor(changeColor)
            Image(systemName: "chevron.right")
                .foregroundColor(.gray)
        }
        .padding(.vertical, 10)
    }
}

// Updated Performance Graph View
struct PerformanceGraphView: View {
    var dataPoints: [CGFloat] = [20, 50, 30, 80, 40, 100, 150]
    
    var body: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 10)
                .fill(Color.clear)
            
            LineGraphView(dataPoints: dataPoints)
                .stroke(Color.green, lineWidth: 2)
                .background(LinearGradient(gradient: Gradient(colors: [Color.green.opacity(0.2), Color.clear]),
                                           startPoint: .top,
                                           endPoint: .bottom)
                    .clipShape(LineGraphView(dataPoints: dataPoints, isBackground: true))
                )
                .padding()
                .overlay(
                    VStack {
                        Spacer()
                        HStack {
                            Text("2020")
                            Spacer()
                            Text("2021")
                            Spacer()
                            Text("2022")
                            Spacer()
                            Text("2023")
                            Spacer()
                            Text("2024")
                        }
                        .padding(.horizontal, 10)
                        .font(.footnote)
                        .foregroundColor(.gray)
                    }
                )
        }
        .frame(height: 200)
        .padding()
    }
}

// Line Graph View to simulate performance
struct LineGraphView: Shape {
    var dataPoints: [CGFloat]
    var isBackground: Bool = false
    
    func path(in rect: CGRect) -> Path {
        var path = Path()
        guard dataPoints.count > 1 else { return path }
        
        let width = rect.width / CGFloat(dataPoints.count - 1)
        let height = rect.height
        
        let maxDataPoint = dataPoints.max() ?? 1
        let minDataPoint = dataPoints.min() ?? 0
        let range = maxDataPoint - minDataPoint
        
        let yFactor = height / range
        
        path.move(to: CGPoint(x: 0, y: height - (dataPoints[0] - minDataPoint) * yFactor))
        
        for index in 1..<dataPoints.count {
            let xPosition = CGFloat(index) * width
            let yPosition = height - (dataPoints[index] - minDataPoint) * yFactor
            path.addLine(to: CGPoint(x: xPosition, y: yPosition))
        }
        
        // Create the gradient fill
        if isBackground {
            path.addLine(to: CGPoint(x: rect.width, y: rect.height))
            path.addLine(to: CGPoint(x: 0, y: rect.height))
            path.closeSubpath()
        }
        
        return path
    }
}

struct SettingsView: View {
    var body: some View {
        Text("Settings")
            .font(.title)
            .padding()
    }
}

// Browse View
struct StockCardView: View {
    var logoName: String
    var symbol: String
    var price: String
    var backgroundColor: Color = Color.black
    var height: CGFloat = 120
    var width: CGFloat = 160
    
    var body: some View {
        VStack {
            Image(logoName)
                .resizable()
                .scaledToFit()
                .frame(width: 40, height: 40) // Adjust size as needed
                .foregroundColor(.white)
            Text(symbol)
                .foregroundColor(.white)
                .bold()
            Text(price)
                .foregroundColor(.white)
                .font(.footnote)
        }
        .padding()
        .frame(width: width, height: height)
        .background(backgroundColor)
        .cornerRadius(20)
    }
}

// Browse View with Search Bar (Front-End)
struct BrowseView: View {
    @State private var searchText = ""
    
    let columns = [
        GridItem(.flexible(), spacing: 20),
        GridItem(.flexible(), spacing: 20)
    ]
    
    var body: some View {
        VStack {
            // Search Bar
            TextField("Search stocks...", text: $searchText)
                .padding()
                .background(Color(.systemGray6))
                .cornerRadius(10)
                .padding(.horizontal)
                .padding(.top, 10) // Add some top padding for better spacing
            
            // Stock Grid
            ScrollView {
                LazyVGrid(columns: columns, spacing: 20) {
                    StockCardView(logoName: "applelogowhite", symbol: "AAPL", price: "$364.11", backgroundColor: Color.black)
                    StockCardView(logoName: "mcdologowhite", symbol: "MCD", price: "$183.52", backgroundColor: Color.red)
                    StockCardView(logoName: "paypallogowhite", symbol: "PYPL", price: "$177.21", backgroundColor: Color.blue)
                    StockCardView(logoName: "lyftlogowhite", symbol: "LYFT", price: "$31.82", backgroundColor: Color.pink)
                    StockCardView(logoName: "amazonlogowhite", symbol: "AMZN", price: "$364.11", backgroundColor: Color.orange)
                    StockCardView(logoName: "nikelogowhite", symbol: "NKE", price: "$98.43", backgroundColor: Color.black)
                    StockCardView(logoName: "googlelogowhite", symbol: "GOOGL", price: "$1,234.56", backgroundColor: Color.green)
                    StockCardView(logoName: "microsoftlogowhite", symbol: "MSFT", price: "$289.67", backgroundColor: Color.purple)
                    StockCardView(logoName: "metalogowhite", symbol: "META", price: "$198.34", backgroundColor: Color.indigo)
                    StockCardView(logoName: "netflixlogowhite", symbol: "NFLX", price: "$533.79", backgroundColor: Color.red)
                    StockCardView(logoName: "teslalogowhite", symbol: "TSLA", price: "$650.21", backgroundColor: Color.gray)
                    StockCardView(logoName: "alibabalogowhite", symbol: "BABA", price: "$211.52", backgroundColor: Color.yellow)
                    StockCardView(logoName: "intellogowhite", symbol: "INTC", price: "$53.79", backgroundColor: Color.blue)
                    StockCardView(logoName: "nvidialogowhite", symbol: "NVDA", price: "$473.50", backgroundColor: Color.teal)
                    StockCardView(logoName: "disneylogowhite", symbol: "DIS", price: "$135.67", backgroundColor: Color.brown)
                    StockCardView(logoName: "tkologowhite", symbol: "TKO", price: "$117.60", backgroundColor: Color.green)
                }
                .padding(.horizontal) // Add horizontal padding
            }
            .padding(.bottom, 10) // Add bottom padding to avoid overlap with other elements
        }
        .background(Color.white.edgesIgnoringSafeArea(.all))
        .frame(maxWidth: .infinity, maxHeight: .infinity) // Make VStack take full available space
    }
}

// SwiftUI Preview
struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
